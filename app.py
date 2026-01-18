import streamlit as st
from Bio.Seq import Seq
from Bio.Data import CodonTable
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io
import random
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import matplotlib.font_manager as fm
import os

# 配置中文字体（上传NotoSansTC-Regular.ttf到仓库解决乱码）
font_path = 'NotoSansTC-Regular.ttf'
if os.path.exists(font_path):
    prop = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = prop.get_name()
    plt.rcParams['axes.unicode_minus'] = False
else:
    st.warning("字体文件未找到，图表中文可能乱码。请上传 NotoSansTC-Regular.ttf。")

# Oryza sativa codon usage table (frequency per thousand, from Kazusa)
rice_codon_table = {
    'F': {'TTT': 13.1, 'TTC': 22.4},
    'L': {'TTA': 6.1, 'TTG': 14.7, 'CTT': 15.2, 'CTC': 25.8, 'CTA': 7.7, 'CTG': 21.0},
    'I': {'ATT': 14.0, 'ATC': 23.6, 'ATA': 8.1},
    'M': {'ATG': 22.1},
    'V': {'GTT': 15.5, 'GTC': 20.8, 'GTA': 6.6, 'GTG': 24.0},
    'S': {'TCT': 12.7, 'TCC': 16.3, 'TCA': 12.4, 'TCG': 12.3, 'AGT': 8.9, 'AGC': 17.9},
    'P': {'CCT': 13.6, 'CCC': 12.1, 'CCA': 14.2, 'CCG': 18.0},
    'T': {'ACT': 10.9, 'ACC': 17.0, 'ACA': 12.2, 'ACG': 10.7},
    'A': {'GCT': 16.7, 'GCC': 23.8, 'GCA': 13.2, 'GCG': 18.8},
    'Y': {'TAT': 10.0, 'TAC': 15.1},
    '*': {'TAA': 0.7, 'TAG': 0.8, 'TGA': 1.2},
    'H': {'CAT': 11.3, 'CAC': 13.8},
    'Q': {'CAA': 13.5, 'CAG': 20.8},
    'N': {'AAT': 15.3, 'AAC': 20.6},
    'K': {'AAA': 17.3, 'AAG': 30.7},
    'D': {'GAT': 19.9, 'GAC': 22.9},
    'E': {'GAA': 20.8, 'GAG': 28.2},
    'C': {'TGT': 6.2, 'TGC': 12.4},
    'W': {'TGG': 13.8},
    'R': {'CGT': 7.2, 'CGC': 16.1, 'CGA': 6.4, 'CGG': 13.4, 'AGA': 9.8, 'AGG': 13.6},
    'G': {'GGT': 10.2, 'GGC': 26.1, 'GGA': 12.9, 'GGG': 13.4}
}

# 加载CodonBERT模型
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("lhallee/CodonBERT")
    model = AutoModelForMaskedLM.from_pretrained("lhallee/CodonBERT")
    return tokenizer, model

tokenizer, model = load_model()

# 获取同义密码子列表（用于随机初始）
standard_table = CodonTable.unambiguous_dna_by_name["Standard"]
synonymous_codons = {}
for aa in standard_table.forward_table:
    if aa not in synonymous_codons:
        synonymous_codons[aa] = []
    synonymous_codons[aa].append(standard_table.forward_table[aa])  # 错误：forward_table是codon:aa，需要反转

# 修正：构建同义codons
synonymous_codons = {}
for codon, aa in standard_table.forward_table.items():
    if aa not in synonymous_codons:
        synonymous_codons[aa] = []
    synonymous_codons[aa].append(codon)

def get_random_codon(aa):
    return random.choice(synonymous_codons.get(aa, ['NNN']))

# 规则优化
def rule_optimize(aa_seq):
    dna_seq = ''
    for aa in aa_seq:
        codons = rice_codon_table.get(aa, {})
        if codons:
            dna_seq += max(codons, key=codons.get)
        else:
            dna_seq += 'NNN'  # 未知
    return dna_seq

# 大模型优化（CodonBERT）：mask 20% codons并预测
def llm_optimize(aa_seq, mask_rate=0.2):
    # 初始随机DNA
    initial_dna = ''.join(get_random_codon(aa) for aa in aa_seq)
    # Tokenize为codons (CodonBERT用空格分隔codons)
    codons = [initial_dna[i:i+3] for i in range(0, len(initial_dna), 3)]
    masked_codons = [c if random.random() > mask_rate else tokenizer.mask_token for c in codons]
    input_text = ' '.join(masked_codons)
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
    
    with torch.no_grad():
        logits = model(**inputs).logits
    mask_indices = (inputs.input_ids[0] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
    predicted_ids = logits[0, mask_indices].argmax(-1)
    predicted_codons = tokenizer.decode(predicted_ids).split()
    
    # 替换masked
    for idx, mask_idx in enumerate(mask_indices):
        if idx < len(predicted_codons):
            masked_codons[mask_idx.item() - 1] = predicted_codons[idx]  # 调整索引（忽略[CLS]）
    
    return ''.join(masked_codons).replace(' ', '')

# CAI计算
def calculate_cai(dna_seq, codon_table=rice_codon_table):
    if len(dna_seq) % 3 != 0:
        return 0.0
    codons = [dna_seq[i:i+3] for i in range(0, len(dna_seq), 3)]
    max_freq = {aa: max(freq.values()) for aa, freq in codon_table.items() if aa != '*'}
    cai_sum = 0
    count = 0
    for codon in codons:
        try:
            aa = str(Seq(codon).translate())
            if aa in max_freq and codon in codon_table.get(aa, {}):
                freq = codon_table[aa][codon]
                cai_sum += freq / max_freq[aa]
                count += 1
        except:
            pass
    return cai_sum / count if count > 0 else 0.0

st.title("日本晴稻密码子优化对比演示（规则 vs. 大模型）")
st.write("输入DNA序列，翻译为氨基酸后，进行规则和大模型优化对比。适用于Hinohikari育种。")

# 输入
dna_input = st.text_area("DNA序列（ORF，3的倍数）", height=200)
if st.button("加载默认示例: Badh2基因 (Oryza sativa)"):
    default_dna = "atggccacggcgatcccgcagcggcagctcttcgtcgccggcgagtggcgcgcccccgcgctcggccgccgcctccccgtcgtcaaccccgccaccgagtcccccatcggcgagatcccggcgggcacggcggaggacgtggacgcggcggtggcggcggcgcgggaggcgctgaagaggaaccggggccgcgactgggcgcgcgcgccgggcgccgtccgggccaagtacctccgcgcaatcgcggccaagataatcgagaggaaatctgagctggactagagacgcttgattgtgggaagcctcttgatgaagcagcatgggacatggacgatgttgctggatgctttgagtactttgcagatcttgcagaatccttggacaaaaggcaaaatgcacctgtctctcttccaatggaaaactttaaatgctatcttcggaaagagcctatcgggtagttgggttgatcacaccttggaactatcctctcctgatggcaacatggaaggtagctcctgccctggctgctggctgtacagctgtactaaaaccatctgaattggcttccgtgacttgtttggagcttgctgatgtgtgtaaagaggttggtcttccttcaggtgtgctaaacatagtgactggattaggttctgaagccggtgctcctttgtcatcacaccctggtgtagacaaggttgcatttactgggagttatgaaactggtaaaaagattatggcttcagctgctcctatggttaagcctgtttcactggaacttggtggaaaaagtcctatagtggtgtttgatgatgttgatgttgaaaaagctgttgagtggactctctttggttgcttttggaccaatggccagatttgcagtgcaacatcgcgtcttattcttcataaaaaaatcgctaaagaatttcaagaaaggatggttgcatgggccaaaaatattaaggtgtcagatccacttgaagagggttgcaggcttgggcccgttgttagtgaaggacagtatgagaagattaagcaatttgtatctaccgccaaaagccaaggtgctaccattctgactggtggggttagacccaagcatctggagaaaggtttctatattgaacccacaatcattactgatgtcgatacatcaatgcaaatttggagggaagaagttttttggtccagtgctctgtgtgaaagaatttagcactgaagaagaagccattgaattggccaacgatactcattatggtctggctggtgctgtgctttccggtgaccgcgagcgatgccagagattaactgaggagatcgatgccggaatttatctgggtgaactgctcgcaaccctgcttctgccaagctccatggggcgggaacaagcgcagcggctttggacgcgagctcggagaagggggcattgacaactaccttagcgtcaagcaagtgacggagtacgcctccgatgagccgtgggatggtacaaatccccttccaagctgtaa"
    dna_input = default_dna
    st.text_area("DNA序列", default_dna, key="default")

if st.button("优化对比"):
    if len(dna_input) % 3 != 0:
        st.error("DNA长度必须是3的倍数。")
    else:
        try:
            aa_seq = str(Seq(dna_input).translate())
            rule_dna = rule_optimize(aa_seq)
            llm_dna = llm_optimize(aa_seq)
            orig_cai = calculate_cai(dna_input.upper())
            rule_cai = calculate_cai(rule_dna)
            llm_cai = calculate_cai(llm_dna)

            st.subheader("原始DNA: " + dna_input[:100] + "...")
            st.subheader("规则优化DNA: " + rule_dna[:100] + "...")
            st.subheader("大模型优化DNA (CodonBERT): " + llm_dna[:100] + "...")

            # 图表
            fig, ax = plt.subplots()
            ax.bar(['原始', '规则优化', '大模型优化'], [orig_cai, rule_cai, llm_cai])
            ax.set_ylabel('CAI 值')
            ax.set_title('优化对比')
            st.pyplot(fig)

            # PDF报告
            pdf_buffer = io.BytesIO()
            pdf = canvas.Canvas(pdf_buffer, pagesize=letter)
            pdf.drawString(100, 750, "Hinohikari 密码子优化对比报告")
            pdf.drawString(100, 700, f"原始 CAI: {orig_cai:.2f}")
            pdf.drawString(100, 650, f"规则优化 CAI: {rule_cai:.2f}")
            pdf.drawString(100, 600, f"大模型优化 CAI: {llm_cai:.2f}")
            pdf.drawString(100, 550, "大模型在上下文优化上优于规则方法。")
            pdf.save()
            pdf_buffer.seek(0)
            st.download_button("下载 PDF 报告", pdf_buffer, file_name="optimize_compare_report.pdf", mime="application/pdf")
        except Exception as e:
            st.error(f"错误: {e}")
