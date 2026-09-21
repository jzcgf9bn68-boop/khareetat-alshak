from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

app = Flask(__name__)
CORS(app)

# ============================================================
# 1) قاعدة الشبهات المُراجَعة يدويًا (الطبقة الأولى)
# ============================================================
CURATED_KB = [
    {
        "claim": "الإسلام انتشر بالسيف وأجبر الناس على الدخول فيه بالقوة",
        "response": "القرآن ينص صراحة: (لا إكراه في الدين قد تبين الرشد من الغي). الجهاد شُرع لرد العدوان لا لإكراه الناس على الدخول بالدين، والتاريخ يُظهر أن أغلب شعوب جنوب شرق آسيا دخلت الإسلام عبر التجارة والدعوة لا الفتح.",
        "source": "القرآن الكريم، سورة البقرة 256",
        "category": "needs_context",
    },
    {
        "claim": "القرآن يأمر بقتل غير المسلمين أينما وُجدوا آية السيف",
        "response": "الآية نزلت في سياق محدد: نقض المشركين لعهودهم مع المسلمين، وتكملة الآية نفسها تشترط وقف القتال عند التوبة وإقامة الصلاة: (فإن تابوا وأقاموا الصلاة وآتوا الزكاة فخلوا سبيلهم).",
        "source": "القرآن الكريم، سورة التوبة 5، مع تفسير ابن كثير",
        "category": "needs_context",
    },
    {
        "claim": "تعدد الزوجات في الإسلام يظلم المرأة",
        "response": "الإسلام قيّد تعددًا كان بلا حدود قبله، بأربع كحد أقصى وشرط العدل التام، ولم يجعله إلزاميًا بل جوازيًا مشروطًا لحالات اجتماعية محددة.",
        "source": "القرآن الكريم، سورة النساء 3",
        "category": "needs_context",
    },
    {
        "claim": "الإسلام يمنع المرأة من التعليم والعمل",
        "response": "لا نص شرعي يمنع ذلك؛ طلب العلم فريضة على كل مسلم ومسلمة. عائشة رضي الله عنها كانت من كبار علماء الصحابة ومرجعًا فقهيًا وحديثيًا.",
        "source": "صحيح البخاري ومسلم، وسير أعلام النبلاء للذهبي",
        "category": "incorrect",
    },
    {
        "claim": "شهادة المرأة نصف شهادة الرجل دليل نقص عقلها",
        "response": "الآية تتحدث تحديدًا عن توثيق الديون المالية، لا عن الشهادة عمومًا، والحكمة المذكورة بالآية نفسها هي احتمال النسيان بمعاملة مالية معقدة، لا وصف بنقص العقل.",
        "source": "القرآن الكريم، سورة البقرة 282",
        "category": "needs_context",
    },
    {
        "claim": "الميراث في الإسلام ظالم لأن المرأة ترث نصف الرجل",
        "response": "هذا صحيح بحالات محددة، لكنه مرتبط بمسؤولية الرجل المالية الكاملة (نفقة الأسرة، المهر) التي لا تلزم المرأة شرعًا، فالمرأة تملك حصتها كاملة بلا التزام إنفاق.",
        "source": "القرآن الكريم، سورة النساء 11",
        "category": "needs_context",
    },
    {
        "claim": "الحجاب أداة قمع للمرأة",
        "response": "الحجاب أمر شرعي يُقصد به الستر والكرامة لا التقييد، وهو اختيار عبادي مبني على نص قرآني صريح لا فرض اجتماعي تعسفي.",
        "source": "القرآن الكريم، سورة النور 31 وسورة الأحزاب 59",
        "category": "needs_context",
    },
    {
        "claim": "الإسلام يبيح العبودية ويشجعها",
        "response": "الإسلام لم يبدأ الرق (كان موجودًا عالميًا قبله)، بل وضع نظامًا تدريجيًا لتحريره: جعل عتق الرقاب كفارة لعشرات الذنوب، وحض عليه كصدقة وقربة، حتى انتهى عمليًا.",
        "source": "القرآن الكريم (النساء 92، المجادلة 3، البلد 13) وأحكام الفقه الإسلامي",
        "category": "needs_context",
    },
    {
        "claim": "الإسلام يحرّم التعايش مع غير المسلمين",
        "response": "القرآن يأمر بالبر والعدل مع غير المحاربين: (لا ينهاكم الله عن الذين لم يقاتلوكم في الدين أن تبروهم وتقسطوا إليهم). عهود أهل الذمة كفلت حماية دينهم وأموالهم.",
        "source": "القرآن الكريم، سورة الممتحنة 8",
        "category": "needs_context",
    },
    {
        "claim": "الرسول محمد ألّف القرآن بنفسه ولم يكن وحيًا من الله",
        "response": "القرآن ينص على أميّة النبي (لا يقرأ ولا يكتب) صراحة: (وما كنت تتلو من قبله من كتاب ولا تخطه بيمينك). يُستشهد بهذا كدليل على استحالة تأليفه نصًا بهذا المستوى البلاغي.",
        "source": "القرآن الكريم، سورة العنكبوت 48",
        "category": "incorrect",
    },
]

CATEGORY_LABELS = {
    "correct": "صحيح وموثق",
    "needs_context": "يحتاج سياق",
    "incorrect": "خاطئ",
}

# ============================================================
# 2) تحميل القرآن والحديث عند بدء تشغيل الخادم
# ============================================================
QURAN_BASE = "https://cdn.jsdelivr.net/gh/fawazahmed0/quran-api@1"
HADITH_BASE = "https://cdn.jsdelivr.net/gh/fawazahmed0/hadith-api@1"

quran_verses = []
all_hadiths = []


def try_direct_editions(base_url, candidates, required_key):
    for c in candidates:
        try:
            r = requests.get(f"{base_url}/editions/{c}.json", timeout=15)
            if r.status_code == 200 and required_key in r.json():
                return c
        except Exception:
            continue
    return None


def load_quran():
    global quran_verses
    try:
        editions = requests.get(f"{QURAN_BASE}/editions.json", timeout=15).json()
        arabic = [k for k in editions.keys() if isinstance(k, str) and k.startswith("ara-")]
    except Exception:
        arabic = []

    edition = None
    preferred = [e for e in arabic if "simple" in e or "uthmani" in e]
    if preferred:
        edition = preferred[0]
    elif arabic:
        edition = arabic[0]
    else:
        edition = try_direct_editions(
            QURAN_BASE,
            ["ara-quransimple", "ara-quranuthmani", "ara-quran", "ara-quran-simple-clean"],
            "quran",
        )

    if not edition:
        print("تحذير: ما قدرت أحمّل القرآن")
        return

    data = requests.get(f"{QURAN_BASE}/editions/{edition}.json", timeout=30).json()
    quran_verses = data["quran"]
    print(f"تم تحميل {len(quran_verses)} آية من إصدار {edition}")


def load_hadiths():
    global all_hadiths
    bukhari_edition = try_direct_editions(HADITH_BASE, ["ara-bukhari"], "hadiths")
    muslim_edition = try_direct_editions(HADITH_BASE, ["ara-muslim"], "hadiths")

    def load_book(edition_name, label):
        if not edition_name:
            return []
        data = requests.get(f"{HADITH_BASE}/editions/{edition_name}.json", timeout=30).json()
        hadiths = data.get("hadiths", [])

        def is_sahih(h):
            grades = h.get("grades") or []
            if not grades:
                return True
            grade_val = str(grades[0].get("grade", "")).lower()
            return grade_val in ("sahih", "صحيح", "")

        sahih_only = [h for h in hadiths if is_sahih(h)]
        return [
            {"book": label, "number": h.get("hadithnumber"), "text": h.get("text", "")}
            for h in sahih_only if h.get("text")
        ]

    all_hadiths = load_book(bukhari_edition, "صحيح البخاري") + load_book(muslim_edition, "صحيح مسلم")
    print(f"تم تحميل {len(all_hadiths)} حديث")


print("جاري تحميل القرآن والحديث... قد يأخذ هذا دقيقة عند أول تشغيل")
load_quran()
load_hadiths()

# ============================================================
# 3) بناء فهارس TF-IDF (خفيفة، تشتغل على الاستضافة المجانية)
# ============================================================
curated_vectorizer = TfidfVectorizer()
curated_matrix = curated_vectorizer.fit_transform([item["claim"] for item in CURATED_KB]) if CURATED_KB else None

if quran_verses:
    quran_vectorizer = TfidfVectorizer()
    quran_matrix = quran_vectorizer.fit_transform([v["text"] for v in quran_verses])
else:
    quran_vectorizer, quran_matrix = None, None

if all_hadiths:
    hadith_vectorizer = TfidfVectorizer()
    hadith_matrix = hadith_vectorizer.fit_transform([h["text"] for h in all_hadiths])
else:
    hadith_vectorizer, hadith_matrix = None, None


def top_matches(query, vectorizer, matrix, k=1):
    if vectorizer is None or matrix is None:
        return []
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, matrix).flatten()
    top_idx = sims.argsort()[-k:][::-1]
    return [(int(i), float(sims[i])) for i in top_idx]


# ============================================================
# 4) نقطة النهاية الرئيسية
# ============================================================
@app.route("/classify", methods=["POST"])
def classify():
    data = request.get_json(force=True)
    claim_text = (data.get("claim") or "").strip()
    if not claim_text:
        return jsonify({"error": "الرجاء إرسال نص الادعاء"}), 400

    curated_threshold = 0.30

    curated_result = top_matches(claim_text, curated_vectorizer, curated_matrix, k=1)
    if curated_result and curated_result[0][1] >= curated_threshold:
        idx, score = curated_result[0]
        match = CURATED_KB[idx]
        return jsonify({
            "tier": "مُراجَع يدويًا",
            "classification": CATEGORY_LABELS[match["category"]],
            "response": match["response"],
            "source": match["source"],
            "confidence": round(score, 3),
        })

    quran_top = top_matches(claim_text, quran_vectorizer, quran_matrix, k=3)
    hadith_top = top_matches(claim_text, hadith_vectorizer, hadith_matrix, k=3)

    related_verses = [
        {
            "reference": f"سورة {quran_verses[i]['chapter']}, آية {quran_verses[i]['verse']}",
            "text": quran_verses[i]["text"],
            "score": round(s, 3),
        }
        for i, s in quran_top
    ]
    related_hadiths = [
        {
            "reference": f"{all_hadiths[i]['book']}, حديث رقم {all_hadiths[i]['number']}",
            "text": all_hadiths[i]["text"],
            "score": round(s, 3),
        }
        for i, s in hadith_top
    ]

    return jsonify({
        "tier": "استرجاع تلقائي - يحتاج مراجعة مختص",
        "classification": "لا يوجد رد مُراجَع جاهز",
        "related_verses": related_verses,
        "related_hadiths": related_hadiths,
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "quran_verses_loaded": len(quran_verses),
        "hadiths_loaded": len(all_hadiths),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
