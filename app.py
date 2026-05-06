from flask import Flask, request, jsonify, render_template

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM



app = Flask(__name__)



# Cache to avoid reloading models repeatedly

MODEL_CACHE = {}



# Custom trained models

CUSTOM_MODELS = {

    ("en", "hi"): "model/my_translation_model"  # keep only if you really have it

}



def get_model(src, tgt):

    if (src, tgt) in MODEL_CACHE:

        return MODEL_CACHE[(src, tgt)]



    if (src, tgt) in CUSTOM_MODELS:

        model_path = CUSTOM_MODELS[(src, tgt)]

    else:

        model_path = f"Helsinki-NLP/opus-mt-{src}-{tgt}"



    try:

        tokenizer = AutoTokenizer.from_pretrained(model_path)

        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

        MODEL_CACHE[(src, tgt)] = (tokenizer, model)

        return tokenizer, model

    except Exception:

        return None, None





@app.route("/")

def home():

    return render_template("index.html")





@app.route("/translate", methods=["POST"])

def translate():

    data = request.get_json()



    text = data.get("text", "").strip()

    src = data.get("source_lang")

    tgt = data.get("target_lang")



    print("SRC:", src, "TGT:", tgt)  # DEBUG (keep for now)



    if not text:

        return jsonify({"translation": ""})



    if src == tgt:

        return jsonify({"translation": text})



    # 1️⃣ Try direct translation

    tokenizer, model = get_model(src, tgt)

    if tokenizer:

        inputs = tokenizer(text, return_tensors="pt", truncation=True)

        outputs = model.generate(**inputs, max_length=128)

        result = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return jsonify({"translation": result})



    # 2️⃣ Pivot via English

    if src != "en" and tgt != "en":

        tok1, mod1 = get_model(src, "en")

        tok2, mod2 = get_model("en", tgt)



        if tok1 and tok2:

            inputs = tok1(text, return_tensors="pt", truncation=True)

            mid = mod1.generate(**inputs, max_length=128)

            mid_text = tok1.decode(mid[0], skip_special_tokens=True)



            inputs = tok2(mid_text, return_tensors="pt", truncation=True)

            out = mod2.generate(**inputs, max_length=128)

            final_text = tok2.decode(out[0], skip_special_tokens=True)



            return jsonify({"translation": final_text})



    return jsonify({"translation": "❌ Language pair not supported yet."})





if __name__ == "__main__":

    app.run(debug=True)