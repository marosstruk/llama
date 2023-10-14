import os

from flask import Flask, request, jsonify, Response
from flask_cors import CORS

from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection
from bson import ObjectId, json_util

from llama import Llama


app = Flask(__name__)
CORS(app)

conn_str = os.environ.get("MONGO_CONN_STR", "mongodb://172.26.0.1:27017")

client = MongoClient(conn_str)
db: Database = client.docparser
coll: Collection = db.docs

# llm = Llama()
alpha_llama = Llama() # Total chad, knows everything, takes his deserved time
beta_llama = Llama() # Lame nerd, only knows one thing, always rushes to please you


@app.route("/api/beta/init", methods=["GET"])
def stimulate_beta_llama():
    doc_id = request.args.get("id")
    doc = coll.find_one({"_id": ObjectId(doc_id)})
    beta_llama.read_documents([doc])
    return Response(status=204)


@app.route("/api/alpha/init", methods=["GET"])
def stimulate_alpha_llama():
    docs = list(coll.find())
    alpha_llama.read_documents(docs)
    return Response(status=204)
    

@app.route("/api/beta/prompt", methods=["POST"])
def beta_submit_prompt():
    prompt = request.data.decode('UTF-8')
    answer = beta_llama.answer_prompt(prompt=prompt)
    return answer['result']


@app.route("/api/alpha/prompt", methods=["POST"])
def alpha_submit_prompt():
    prompt = request.data.decode('UTF-8')
    answer = alpha_llama.answer_prompt(prompt=prompt)
    return answer['result']


if __name__ == "__main__":
    #os.environ["MONGO_CONN_STR"] = "mongodb://172.26.0.1:27017"
    app.run(debug=True)