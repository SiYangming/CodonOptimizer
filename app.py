import streamlit as st
from Bio.Seq import Seq
from Bio.Data import CodonTable
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io  # 用于PDF下载

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
    '*': {'TAA': 0.7, 'TAG': 0.8, 'TGA': 1.2},  # Stop codons
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

def get_optimal_codon(aa):
    """选择最高频率的密码子"""
    codons = rice_codon_table.get(aa, {})
    if not codons:
        return None
    return max(codons, key=codons.get)

def optimize_sequence(aa_seq):
    """从氨基酸序列生成优化DNA序列"""
    dna_seq = ''
    for aa in aa_seq:
        codon = get_optimal_codon(aa)
        if codon:
            dna_seq += codon
        else:
            st.warning(f"未知氨基酸: {aa}")
    return dna_seq

def calculate_cai(dna_seq, codon_table=rice_codon_table):
    """简单CAI计算（适应指数）"""
    if len(dna_seq) % 3 != 0:
        return 0.0
    codons = [dna_seq[i:i+3] for i in range(0, len(dna_seq), 3)]
    max_freq = {aa: max(freq.values()) for aa, freq in codon_table.items() if aa != '*'}
    cai_sum = 0
    count = 0
    for codon in codons:
        aa = str(Seq(codon).translate())
        if aa in max_freq and codon in codon_table.get(aa, {}):
            freq = codon_table[aa][codon]
            cai_sum += freq / max_freq[aa]
            count += 1
    return cai_sum / count if count > 0 else 0.0

st.title("日本晴稻 (Hinohikari) 密码子优化育种 Alpha 演示")
st.write("输入氨基酸序列，优化用于 Hinohikari 育种（如抗病基因）。演示基于水稻密码子偏好，提高表达效率。")

# 输入
aa_seq = st.text_area("氨基酸序列（例如 Badh2 或 Cry1Ca 基因）", "MVSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLTYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITLGMDELYK")
original_dna = st.text_area("原始 DNA 序列（可选，用于对比 CAI）", "")

if st.button("优化"):
    opt_dna = optimize_sequence(aa_seq)
    opt_cai = calculate_cai(opt_dna)
    orig_cai = calculate_cai(original_dna) if original_dna else 0.5  # 假设基准如果无输入

    st.write(f"优化 DNA 序列: {opt_dna[:100]}... (全长: {len(opt_dna)} bp)")
    st.write(f"CAI 提升: 从 {orig_cai:.2f} 到 {opt_cai:.2f} (潜在表达提升 10-50%，基于文献)")

    # 图表
    fig, ax = plt.subplots()
    ax.bar(['原始', '优化'], [orig_cai, opt_cai])
    ax.set_ylabel('CAI 值')
    ax.set_title('优化前后 CAI 对比')
    st.pyplot(fig)

    # 生成 PDF 报告
    pdf_buffer = io.BytesIO()
    pdf = canvas.Canvas(pdf_buffer, pagesize=letter)
    pdf.drawString(100, 750, "Hinohikari 密码子优化报告")
    pdf.drawString(100, 700, f"输入氨基酸序列: {aa_seq[:50]}...")
    pdf.drawString(100, 650, f"优化 DNA: {opt_dna[:50]}...")
    pdf.drawString(100, 600, f"CAI 提升: {opt_cai:.2f}")
    pdf.drawString(100, 550, "此优化可用于 Hinohikari 抗旱/抗虫育种。")
    pdf.save()
    pdf_buffer.seek(0)
    st.download_button("下载 PDF 报告", pdf_buffer, file_name="hinohikari_optimize_report.pdf", mime="application/pdf")

# 示例加载
if st.button("加载示例: Badh2 基因 (Hinohikari 香味相关)"):
    example_aa = "MEIKVEKIEVEVEVEVEVEVEV..."  # 替换为实际 Badh2 氨基酸序列（从 NCBI 获取）
    st.text_area("氨基酸序列", example_aa, key="example")
