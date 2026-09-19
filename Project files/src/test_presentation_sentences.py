import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, os.path.abspath('src'))
sys.path.insert(0, os.path.abspath('.'))
from inference_pipeline import UnifiedBanglaTextAnalyzer

pipeline = UnifiedBanglaTextAnalyzer()

test_sentences = [
    # 1. Clear Positive
    ('এই বইটির গল্প এবং লেখার মান অসাধারণ, পড়ে মুগ্ধ হলাম!', 'Positive & Non-Sarcastic & Clean'),
    ('খাবারটা সত্যিই অনেক সুস্বাদু ছিল এবং সার্ভিসও খুব চমৎকার।', 'Positive & Non-Sarcastic & Clean'),
    
    # 2. Clear Negative
    ('খাবারটা অত্যন্ত বাজে এবং সার্ভিস খুবই বিরক্তিকর ছিল।', 'Negative & Non-Sarcastic & Clean'),
    ('এত জঘন্য ডেলিভারি এবং খারাপ ব্যবহার আমি কোথাও দেখিনি।', 'Negative & Non-Sarcastic & Clean'),
    
    # 3. Neutral / Factual
    ('আমি ভাত খাই।', 'Neutral & Non-Sarcastic & Clean'),
    ('গতকাল বিকাল ৫টায় আমাদের ডিপার্টমেন্টের মিটিং শেষ হয়েছে।', 'Neutral & Non-Sarcastic & Clean'),
    ('বইটির দাম ৩৫০ টাকা এবং এতে মোট ১২০ পৃষ্ঠা আছে।', 'Neutral & Non-Sarcastic & Clean'),
    
    # 4. Sarcastic Inversion (Our Core Research Contribution!)
    ('বাহ! কী অসাধারণ service, তিন ঘণ্টা অপেক্ষা করেও কাজ হলো না!', 'Negative (Inverted) & Sarcastic & Clean'),
    ('অসাধারণ! পরীক্ষার ঠিক আগের দিন বিদ্যুৎ চলে গেল, খুব সুন্দর হলো!', 'Negative (Inverted) & Sarcastic & Clean'),
    
    # 5. Hate Speech vs Clean Assertive
    ('তুই একটা অপদার্থ জারজ, তোদের দেশ থেকে তাড়ানো উচিত।', 'Negative & Non-Sarcastic & Hate Speech'),
    ('আপনার এই সিদ্ধান্তের সাথে আমি দ্বিমত পোষণ করছি, তবে আলোচনা করা যায়।', 'Neutral/Non-Hate Clean')
]

print('='*80)
for s, expected in test_sentences:
    res = pipeline.analyze(s, model_type='ensemble')
    sent = res['sentiment']
    sarc = res['sarcasm']
    hate = res['hate_speech']
    inv = ' [Inverted]' if sent.get('context_inverted') else ''
    l_sent = sent['label']
    c_sent = sent['confidence']
    l_sarc = sarc['label']
    c_sarc = sarc['confidence']
    l_hate = hate['label']
    c_hate = hate['confidence']
    print(f'Text: {s}')
    print(f'  Target Expected: {expected}')
    print(f'  -> Sentiment : {l_sent} ({c_sent}%) {inv}')
    print(f'  -> Sarcasm   : {l_sarc} ({c_sarc}%)')
    print(f'  -> HateSpeech: {l_hate} ({c_hate}%)')
    print('-'*80)
