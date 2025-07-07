from flask import Flask, request, jsonify
from flask_cors import CORS
import csv
from difflib import get_close_matches
from collections import Counter
from openai import OpenAI

app = Flask(__name__)
CORS(app)  # 모든 도메인에서의 요청 허용

# OpenAI 클라이언트 생성
client = OpenAI(api_key="Key")  # 실제 키는 보안상 숨기기

# 1. CSV 로딩 (dialect: standard, region)
def load_dialect_dictionary(csv_path):
    mapping = {}
    region_map = {}
    with open(csv_path, encoding='cp949') as f:
        reader = csv.reader(f)
        next(reader)  # 헤더 스킵
        for row in reader:
            if len(row) >= 2:
                dialect = row[0].strip()
                standard = row[1].strip()
                region = row[2].strip() if len(row) > 2 else ""
                mapping[dialect] = standard
                region_map[dialect] = region
    return mapping, region_map

# 2. 단어 유사도 변환
def convert_word(word, mapping, cutoff=0.6):
    candidates = get_close_matches(word, mapping.keys(), n=1, cutoff=cutoff)
    if candidates:
        return mapping[candidates[0]], candidates[0], False
    return word, None, True

# 3. 문장 변환 + 일치한 사투리 단어 리스트도 반환
def convert_sentence(sentence, mapping):
    words = sentence.split()
    converted_words = []
    dialect_words = []
    trigger = False

    for word in words:
        converted, dialect_key, failed = convert_word(word, mapping)
        converted_words.append(converted)
        if dialect_key:
            dialect_words.append(dialect_key)
        if failed:
            trigger = True

    return ' '.join(converted_words), trigger, dialect_words

# 4. 지역 추론 함수
def infer_region(dialect_words, region_map):
    regions = [region_map.get(word) for word in dialect_words if word in region_map]
    if not regions:
        return "알 수 없음"
    return Counter(regions).most_common(1)[0][0]

# 5. GPT fallback
def gpt_translate(sentence):
    prompt = f"다음 문장은 사투리야. 이 문장을 표준어로 바꿔줘:\n\"{sentence}\""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "당신은 사투리 번역기입니다."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5
    )
    return response.choices[0].message.content.strip().replace('\"', '')

# 6. 전역 매핑 로드
mapping, region_map = load_dialect_dictionary("dialect_dict.csv")

@app.route('/translate', methods=['POST'])
def translate():
    data = request.get_json()
    if not data or 'sentence' not in data:
        return jsonify({'error': '문장(sentence) 필드가 필요합니다.'}), 400

    sentence = data['sentence']
    converted, trigger, dialect_words = convert_sentence(sentence, mapping)
    region = infer_region(dialect_words, region_map)

    if trigger:
        print("[!] 일부 단어를 찾지 못해 GPT를 사용합니다.")
        converted = gpt_translate(sentence)

    return jsonify({
        'original': sentence,
        'converted': converted,
        'region': region
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
