# bloom_dedup.py
import os
import hashlib
from datetime import datetime
from pathlib import Path

from pybloom_live import ScalableBloomFilter

from src.core import config as cfg


class BloomTextDedup:
    """
    基于布隆过滤器的文本去重工具，支持持久化存储
    """

    def __init__(self,
                 db_path=os.path.join(cfg.bloom_folder, 'bloom_dedup.bin'),
                 initial_capacity=100000,
                 error_rate=0.001):
        """
        初始化布隆过滤器

        Args:
            db_path: 持久化文件路径
            initial_capacity: 初始容量（预计存储的元素数量）
            error_rate: 误判率（0.001 = 0.1%）
        """
        Path(cfg.bloom_folder).mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self.initial_capacity = initial_capacity
        self.error_rate = error_rate
        self.stats = {
            'total_added': 0,
            'total_duplicates': 0,
            'created_at': None,
            'last_updated': None
        }

        # 加载或创建布隆过滤器
        if os.path.exists(db_path):
            self._load()
        else:
            self._create()

    def _create(self):
        """创建新的布隆过滤器"""
        self.bloom = ScalableBloomFilter(
            initial_capacity=self.initial_capacity,
            error_rate=self.error_rate,
            mode=ScalableBloomFilter.SMALL_SET_GROWTH
        )
        self.stats['created_at'] = datetime.now().isoformat()
        self.stats['last_updated'] = self.stats['created_at']
        print(f"✓ 创建新的布隆过滤器，容量={self.initial_capacity}, 误判率={self.error_rate}")

    def _load(self):
        """从文件加载布隆过滤器"""
        try:
            with open(self.db_path, 'rb') as f:
                self.bloom = ScalableBloomFilter.fromfile(f)
            # 加载统计信息（如果有）
            meta_path = self.db_path + '.meta'
            if os.path.exists(meta_path):
                import json
                with open(meta_path, 'r', encoding='utf-8') as f:
                    self.stats = json.load(f)
            print(f"✓ 已加载布隆过滤器，当前元素数≈{len(self.bloom)}")
        except Exception as e:
            print(f"⚠ 加载失败: {e}，将创建新的过滤器")
            self._create()

    def _save(self):
        """保存到文件"""
        with open(self.db_path, 'wb') as f:
            self.bloom.tofile(f)

        # 保存统计信息
        meta_path = self.db_path + '.meta'
        import json
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2, ensure_ascii=False)

    def add(self, text):
        """
        添加文本，返回是否为新文本

        Returns:
            bool: True=新文本，False=重复文本
        """
        if not text or not text.strip():
            return False

        # 使用哈希值存储，节省内存
        text_hash = self._hash_text(text)

        if text_hash in self.bloom:
            self.stats['total_duplicates'] += 1
            return False  # 重复
        else:
            self.bloom.add(text_hash)
            self.stats['total_added'] += 1
            self.stats['last_updated'] = datetime.now().isoformat()
            return True  # 新文本

    def add_batch(self, texts, auto_save=True):
        """
        批量添加文本

        Args:
            texts: 文本列表
            auto_save: 是否自动保存

        Returns:
            dict: 统计信息
        """
        new_count = 0
        dup_count = 0

        for text in texts:
            if self.add(text):
                new_count += 1
            else:
                dup_count += 1

        if auto_save:
            self.save()

        return {
            'total': len(texts),
            'new': new_count,
            'duplicates': dup_count,
            'duplicate_rate': f"{dup_count / len(texts) * 100:.2f}%" if texts else "0%"
        }

    def contains(self, text):
        """
        检查文本是否已存在

        Returns:
            bool: True=可能存在（有误判率），False=肯定不存在
        """
        if not text or not text.strip():
            return False
        text_hash = self._hash_text(text)
        return text_hash in self.bloom

    def _hash_text(self, text):
        """将文本转换为哈希值（节省内存）"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def save(self):
        """手动保存"""
        self._save()
        print(f"✓ 已保存，总添加={self.stats['total_added']}, 重复={self.stats['total_duplicates']}")

    def get_stats(self):
        """获取统计信息"""
        return {
            **self.stats,
            'current_size': len(self.bloom),
            'file_path': self.db_path,
            'file_size': f"{os.path.getsize(self.db_path) / 1024:.2f} KB" if os.path.exists(self.db_path) else "N/A"
        }

    def reset(self):
        """重置布隆过滤器"""
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        if os.path.exists(self.db_path + '.meta'):
            os.remove(self.db_path + '.meta')
        self._create()
        print("✓ 已重置布隆过滤器")


# ==================== 使用示例 ====================

if __name__ == '__main__':
    import time

    print("=" * 60)
    print("Bloom Filter 文本去重演示")
    print("=" * 60)

    # 1. 创建去重器
    dedup = BloomTextDedup(
        db_path='my_dedup.bin',
        initial_capacity=100000,
        error_rate=0.001
    )

    # 2. 单条添加测试
    print("\n【单条添加测试】")
    texts = ["你好", "世界", "你好", "Python", "世界", "去重测试"]
    for text in texts:
        is_new = dedup.add(text)
        status = "✓ 新文本" if is_new else "✗ 重复"
        print(f"  {text:10} -> {status}")

    # 3. 批量添加测试
    print("\n【批量添加测试】")
    batch_texts = [f"文本_{i}" for i in range(1000)] * 2  # 2000条，50%重复
    start = time.time()
    result = dedup.add_batch(batch_texts)
    elapsed = time.time() - start
    print(f"  处理 {result['total']} 条文本")
    print(f"  新文本: {result['new']}")
    print(f"  重复: {result['duplicates']} ({result['duplicate_rate']})")
    print(f"  耗时: {elapsed:.3f} 秒")
    print(f"  速度: {result['total'] / elapsed:.0f} 条/秒")

    # 4. 查看统计
    print("\n【统计信息】")
    stats = dedup.get_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # 5. 持久化验证
    print("\n【持久化验证】")
    dedup.save()

    # 重新加载
    dedup2 = BloomTextDedup(db_path='my_dedup.bin')
    print(f"  重新加载后检查 '你好': {'存在' if dedup2.contains('你好') else '不存在'}")
    print(f"  重新加载后检查 '新文本': {'存在' if dedup2.contains('新文本') else '不存在'}")

    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)
