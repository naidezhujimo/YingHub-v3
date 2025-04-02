import json
import re
from glob import glob
from tqdm import tqdm
import numpy as np
import random

config = {
    "input_dirs": {
        "gutenberg": "./raw_texts/gutenberg.jsonl",
        "shakespeare": "./raw_texts/shakespeare.txt"
    },
    "output_file": "./cleaned_data.txt",
    "special_tokens": {
        "sep": "[SEP]", # 段落分隔符
        "act": "<ACT>", # ACT（幕）的标记
        "scene": "<SCENE>", # SCENE（场景）的标记
        "stanza": "[STANZA]", # STANZA（诗节）的标记
        "line": "[LINE]", # LINE（行）的标记
        "speaker_start": "<SPEAKER>", # 角色对话的起始和结束标记
        "speaker_end": "</SPEAKER>",
        "stage_start": "<STAGE>", # 舞台指示的起始和结束标记
        "stage_end": "</STAGE>"
    }
}

def process_gutenberg(filepath):
    """处理古登堡JSON数据，增强段落合并逻辑"""
    paragraphs = []
    with open(filepath, 'r') as f:
        current_gid = None
        buffer = []
        mix_ratio = 0.1  # 古登堡占比10%
        for line in f: # 每行是一个JSON对象，包含gid（段落ID）和s（段落内容）
            if random.random() < mix_ratio:
                try:
                    data = json.loads(line)
                    if data["gid"] != current_gid and buffer: # 如果当前行的gid与上一行不同，且缓冲区buffer中有内容，则将缓冲区中的段落合并为一个段落
                        merged_para = " ".join(buffer).replace("  ", " ")
                        if len(merged_para) > 50: # 合并后的段落如果长度大于50，则加入到结果列表中
                            paragraphs.append(merged_para) # 将当前行的内容加入缓冲区
                        buffer = []
                    current_gid = data["gid"]
                    buffer.append(data["s"].strip())
                except json.JSONDecodeError:
                    continue
        
        if buffer: # 最后，如果缓冲区中还有内容，将其合并为一个段落并加入结果列表。
            paragraphs.append(" ".join(buffer))
    return paragraphs

def clean_shakespeare(text):
    """莎士比亚文本清洗主函数"""
   # 替换<ACT>标签为配置文件中定义的特殊标记
    text = text.replace('<ACT>', config["special_tokens"]["act"])
    
    # 新增场景标注处理
    text = re.sub(
        r'SCENE ([IVXLCDM]+)\.\s*(.+)',
        f'\n{config["special_tokens"]["scene"]}\\1: \\2</SCENE>\n',
        text,
        flags=re.IGNORECASE|re.MULTILINE
    )

    # 阶段1: 预处理特殊符号
    text = re.sub(
        r'_+([^_]+?)_+',
        lambda m: f'[UNDERSCORE]{m.group(1).strip("_")}[/UNDERSCORE]',
        text
    )

    # 阶段2: ACT标签处理（保持原逻辑）
    roman_numeral = r'M{{0,4}}(CM|CD|D?C{{0,3}})(XC|XL|L?X{{0,3}})(IX|IV|V?I{{0,3}})'
    text = re.sub(
        rf'ACT ({roman_numeral})\b',
        f'\n{config["special_tokens"]["act"]}\\1</ACT>\n',
        text,
        flags=re.IGNORECASE|re.MULTILINE
    )

    # 强化角色标注处理
    text = re.sub(
        r'\n([A-Z][A-Z]+\b[\sA-Z]*)[\.\s]*\n(?![^<]*</SPEAKER>)',  # 确保后续无闭合标签
        lambda m: f'\n{config["special_tokens"]["speaker_start"]}{m.group(1).rstrip(". ")}{config["special_tokens"]["speaker_end"]}\n',
        text,
        flags=re.IGNORECASE
    )

    # 新增场景切换后的空行处理
    text = re.sub(r'(</SCENE>)\n(?=\w)', r'\1\n\n', text)

    # 阶段3: 舞台指示处理
    # 修改为（精确匹配开闭标签）：
    stage_patterns = [
        (r'(\[Exeunt\.?\])(</SCENE>)?', r'\1'),  # 允许可选闭合标签
        (r'\[UNDERSCORE\](.*?)\[/UNDERSCORE\]', r'\1'),
        (r'^(Enter [A-Za-z].+?)(?=\n\[LINE\]|$)', r'\1'),
        (r'(Exeunt\.?)$', r'\1'),
        (r'(\[.*?\])', r'\1'),
        (r'^(Exit [A-Za-z].+?\.?)(?=\n|\Z)', r'\1')
    ]

    for pattern, replacement in stage_patterns:
        text = re.sub(
            pattern,
            f'{config["special_tokens"]["stage_start"]}{replacement}{config["special_tokens"]["stage_end"]}',
            text,
            flags=re.MULTILINE|re.DOTALL
        )

    # 阶段4: 处理嵌套结构
    text = re.sub(
        r'<STAGE>(.*?)<STAGE>(.*?)</STAGE>(.*?)</STAGE>',
        lambda m: f'{config["special_tokens"]["stage_start"]}{m.group(1)}{m.group(2)}{m.group(3)}{config["special_tokens"]["stage_end"]}',
        text,
        flags=re.DOTALL
    )

    # 修复ACT标签被错误包裹在SPEAKER中的问题
    text = re.sub(
        r'<SPEAKER>ACT (\w+)</SPEAKER>',
        f'\n{config["special_tokens"]["act"]}\\1</ACT>\n',
        text
    )


    # 修改舞台指示嵌套处理（增强版）
    text = re.sub(
        r'<STAGE>(\[?<STAGE>.*?</STAGE>]?)</STAGE>',
        lambda m: f'{config["special_tokens"]["stage_start"]}{re.sub(r"</?STAGE>", "", m.group(1))}{config["special_tokens"]["stage_end"]}',
        text,
        flags=re.DOTALL
    )
    
    # 阶段5: 最终清理
    text = re.sub(r'\[/?UNDERSCORE\]', '', text)
    text = re.sub(
        r'<STAGE>(.*?)[._](</STAGE>)',
        lambda m: f'{config["special_tokens"]["stage_start"]}{m.group(1).strip("._")}{m.group(2)}',
        text
    )
    text = re.sub(r"<(\w+)(.*?)>(?!.*<\/\1>)", r"<\1\2></\1>", text)  # 补全缺失的闭合标签
    
    # 新增：移除重复的闭合标签（如</SPEAKER></SPEAKER>）
    text = re.sub(
        r'(</(SPEAKER|STAGE|SCENE)>)\s*</\2>',
        r'</\2>',
        text,
        flags=re.IGNORECASE
    )
    
    return re.sub(r'\n{3,}', '\n\n', text)

def normalize_archaic(text):
    """古英语标准化处理"""
    replacements = {
        r"\bo['’]\s*th['’]\b": "of the ",
        r"\bYonder['’]s\b": "Yonder is",
        r"\b(['’])T\b": "t",
        r"\b(['’])tis\b": "it is",
        r"\bcannot\b": "can not",
        r"\bi['’]\s*th['’]\b": "in the ",
        r"\b(['’])em\b": "them",
        r"\b(['’])Twas\b": "It was",
        r"\b([Ff])or’s\b": r"\1or his",     # for's -> for his
        r"\b([Tt])is\b": r"\1is",          # 处理独立tis
        r"\b([Ww])ere’t\b": r"\1ere it",   # were’t -> were it
        r"\b([Ll])ord’s\b": r"\1ord's",    # 保留所有格
        r"\b’([A-Z])": r"'\1",             # 处理首字母大写的缩写
    }
    replacements.update({
        r"\b([Hh])ath\b": r"\1as",  # hath -> has
        r"\b([Tt])'?is\b": r"\1t is",  # 'tis -> it is
        r"\b([Ww])ere’t\b": r"\1ere it",  # were’t -> were it
        r"\b([Ee])’er\b": r"\1ver",  # e'er -> ever
    })
    for pattern, repl in replacements.items():
        text = re.sub(pattern, repl, text)
    return text

def process_all_data():
    all_paragraphs = []
    
    # 古登堡数据处理 
    for filepath in tqdm(glob(config["input_dirs"]["gutenberg"]), desc="Processing Gutenberg"):
        all_paragraphs.extend(process_gutenberg(filepath))
    
    # 莎士比亚数据处理
    for filepath in tqdm(glob(config["input_dirs"]["shakespeare"]), desc="Processing Shakespeare"):
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
            cleaned = clean_shakespeare(text)
            paragraphs = [
                p.strip() 
                for p in cleaned.split(f'\n{config["special_tokens"]["stanza"]}\n') 
                if p.strip()
            ]
            all_paragraphs.extend(paragraphs)
    
    # 统一后处理
    processed = []
    for para in all_paragraphs:
        para = re.sub(r'\^[0-9]+', '', para)  # 脚注
        para = re.sub(r'\*+([^\*]+)\*+', r'<EMPHASIS>\1</EMPHASIS>', para)
        para = normalize_archaic(para)
        para = re.sub(r'[^\x00-\x7F]+', '', para)  # 移除非ASCII字符（如中文）
        # 清理格式残留
        para = re.sub(r'\n+\[LINE\]', f'\n{config["special_tokens"]["line"]}', para)
        para = re.sub(r'\[LINE\]\s+\[LINE\]', config["special_tokens"]["line"], para)
        
        # # 新增：随机句子置换
        # if random.random() < 0.3:
        #     sentences = re.split(r'(?<=[.!?]) +', para)
        #     random.shuffle(sentences)
        #     para = ' '.join(sentences)
        
        # # 新增：动态遮蔽
        # if random.random() < 0.2:
        #     words = para.split()
        #     mask_idx = random.sample(range(len(words)), k=int(len(words)*0.1))
        #     words = [w if i not in mask_idx else '[MASK]' for i,w in enumerate(words)]
        #     para = ' '.join(words)
        
        para = re.sub(r'[^\x00-\x7F]+', '', para) 
        
        processed.append(
            re.sub(r'^\s{4,}', '', para)
            .replace('\t', ' ')
            .strip()
        )

    
    # 最终输出处理
    np.random.shuffle(processed)
    with open(config["output_file"], 'w', encoding='utf-8') as f:
        content = f'\n{config["special_tokens"]["sep"]}\n'.join(processed)
        content = re.sub(
            fr'({re.escape(config["special_tokens"]["line"])}\n)+', 
            f'{config["special_tokens"]["line"]}\n', 
            content
        )
        content = re.sub(r'<(\w+)></\1>', '', content)
        f.write(re.sub(r' {2,}', ' ', content))

if __name__ == "__main__":
    process_all_data()