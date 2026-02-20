import requests


def print_line():
    print("-" * 100)


# response = requests.post("http://localhost:8000/chat",
#                          headers={"Content-Type": "application/json"},
#                          json={"messages": ["你好啊！"]})
# print(response.json().get("response"))

print_line()
response = requests.post("http://localhost:8000/rag",
                         headers={"Content-Type": "application/json"},
                         json={"query": "电话是多少？我没有你们的货币怎么办？"})
print(response.json().get("response"))
