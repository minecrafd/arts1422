from flask import Flask, request

app = Flask(__name__)

@app.route("/", methods=["POST"])
def process_data():
    if request.method == "POST":
        bv_value = request.form.get("bv")
        # 在这里执行您的操作，使用获取到的 bv_value

        return "Success"  # 返回响应

if __name__ == "__main__":
    app.run(port=8080)
