import getpass
import os

if not os.environ.get("GOOGLE_API_KEY"):
  os.environ["GOOGLE_API_KEY"] = ""

from langchain.chat_models import init_chat_model

model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")

from langchain_core.messages import HumanMessage

# response = model.invoke([HumanMessage(content="hi!")])
# response.content

with open('Customer1.json', 'r') as f:
    customer1 = f.read()

response = model.invoke([HumanMessage(content="Analyze the following customer data and summarize key insights. The transaction is being flagged and an official will be investigating it using your report. Make sure to give proper markdown format. Customer data:\n\n" + customer1)])

# print(response.content)
