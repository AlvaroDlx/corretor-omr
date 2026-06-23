import streamlit as st
import cv2
import numpy as np
import pandas as pd

# ==========================================
# CONFIGURAÇÃO DE INTERFACE DA PLATAFORMA
# ==========================================
st.set_page_config(
    page_title="Cognitive OMR Vision Engine v6.0", 
    page_icon="🧠", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
        .block-container { padding-top: 2rem; }
        .stMetric { background-color: #f8f9fa; padding: 15px; border-radius: 10px; border: 1px solid #e9ecef; }
        .stAlert { border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# MOTOR DE INTELIGÊNCIA ARTIFICIAL E PROPORÇÃO
# ==========================================
class OMRProcessingEngine:

    @staticmethod
    def normalize_resolution(image_bytes, target_width=1000):
        """Garante consistência absoluta: qualquer resolução vira 1000px de largura."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Formato de imagem inválido.")
        h, w = img.shape[:2]
        scale = target_width / float(w)
        return cv2.resize(img, (target_width, int(h * scale)), interpolation=cv2.INTER_AREA), scale

    @classmethod
    def process_form(cls, image_bytes, num_options):
        """Processamento independente baseado em geometria matricial adaptativa."""
        src, scale = cls.normalize_resolution(image_bytes)
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        
        # Filtro de binarização adaptativa otimizado para remover ruídos e texturas de fundo
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 41, 7
        )
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bubbles = []
        
        # Filtragem geométrica baseada na proporção de escala da largura de 1000px
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h)
            
            # Na escala de 1000px, caixas de questões e alternativas medem entre 25 e 85 pixels
            if 25 <= w <= 85 and 25 <= h <= 85 and 0.7 <= aspect_ratio <= 1.3:
                area = cv2.contourArea(cnt)
                extent = area / float(w * h)
                
                # Filtra assinaturas, textos corridos ou linhas finas de tabelas
                if extent > 0.35:
                    mask = np.zeros(thresh.shape, dtype="uint8")
                    cv2.drawContours(mask, [cnt], -1, 255, -1)
                    density = cv2.mean(thresh, mask=mask)[0]
                    
                    bubbles.append({
                        'x': x, 'y': y, 'w': w, 'h': h,
                        'cx': x + w / 2.0, 'cy': y + h / 2.0,
                        'density': density, 'cnt': cnt
                    })
                    
        if not bubbles:
            return [], src, thresh, "Nenhum elemento estrutural foi identificado na folha."

        # AGRUPAMENTO GEOGRÁFICO DE LINHAS (Pensamento adaptativo por proximidade)
        rows = []
        bubbles.sort(key=lambda b: b['cy'])
        
        for b in bubbles:
            inserted = False
            for row in rows:
                row_cy = np.mean([item['cy'] for item in row])
                # Agrupa na mesma linha se a variação vertical for mínima
                if abs(b['cy'] - row_cy) < 28:
                    row.append(b)
                    inserted = True
                    break
            if not inserted:
                rows.append([b])
                
        # Ordena as linhas de cima para baixo
        rows.sort(key=lambda r: np.mean([item['cy'] for item in r]))
        # Filtra ruídos isolados que não formam uma linha de gabarito (mínimo número + 1 alternativa)
        rows = [r for r in rows if len(r) >= 2]
        
        # CÁLCULO DINDÂMICO DO ESPAÇAMENTO LATERAL
        spacings = []
        for r in rows:
            r.sort(key=lambda b: b['cx'])
            if len(r) >= 3:
                for i in range(len(r) - 1):
                    spacings.append(r[i+1]['cx'] - r[i]['cx'])
                    
        avg_spacing = np.median(spacings) if spacings else 75
        
        options_alphabet = ['A', 'B', 'C', 'D', 'E', 'F'][:num_options]
        parsed_results = []
        
        for idx, r in enumerate(rows):
            # Garante a ordenação da esquerda para a direita
            r.sort(key=lambda b: b['cx'])
            
            # DEFINIÇÃO DA ÂNCORA: O primeiro elemento à esquerda é o número da questão!
            q_box = r[0]
            alternative_boxes = r[1:]
            
            marked_choice = "NULO"
            highest_density = -1
            chosen_box = None
            
            for b in alternative_boxes:
                # Calcula a distância em relação ao número da questão para saber a letra exata
                distance_from_anchor = b['cx'] - q_box['cx']
                detected_col = int(round(distance_from_anchor / avg_spacing)) - 1
                
                if 0 <= detected_col < num_options:
                    # Avalia preenchimento (limiar de corte adaptativo para caneta/grafite)
                    if b['density'] > highest_density and b['density'] > 45:
                        highest_density = b['density']
                        marked_choice = options_alphabet[detected_col]
                        chosen_box = b
                        
            parsed_results.append({
                'question_index': idx + 1,
                'q_box': q_box,
                'choice': marked_choice,
                'chosen_box': chosen_box,
                'all_components': r
            })
            
        return parsed_results, src, thresh, None

# ==========================================
# INTERFACE GRÁFICA DO USUÁRIO (STREAMLIT)
# ==========================================
def main():
    st.sidebar.title("🧠 Inteligência OMR")
    num_options = st.sidebar.selectbox("Alternativas por Questão", [5, 4], format_func=lambda x: f"{x} Opções (A até {chr(64+x)})")
    
    st.title("🧠 Corretor Cognitivo de Gabaritos")
    st.write("Insira os cartões de resposta. O motor identificará as caixas dos números e as marcações de forma autônoma.")
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("📁 Gabarito Oficial (Mestre)")
        master_file = st.file_uploader("Upload do gabarito oficial", type=['jpg','jpeg','png'], key="master")
    with c2:
        st.subheader("📁 Cartão de Respostas (Aluno)")
        student_file = st.file_uploader("Upload da folha do aluno", type=['jpg','jpeg','png'], key="student")
        
    if master_file and student_file:
        st.markdown("---")
        if st.button("🚀 Processar e Corrigir Ambas as Imagens", use_container_width=True, type="primary"):
            with st.spinner("Executando leitura matricial estruturada..."):
                
                m_res, m_img, _, m_err = OMRProcessingEngine.process_form(master_file.getvalue(), num_options)
                s_res, s_img, _, s_err = OMRProcessingEngine.process_form(student_file.getvalue(), num_options)
                
                if m_err or s_err:
                    st.error(f"Não foi possível processar: {m_err if m_err else s_err}")
                    return
                
                # Cruzamento de dados inteligente baseado no número real mapeado
                master_dict = {item['question_index']: item['choice'] for item in m_res}
                
                correct_count = 0
                table_report = []
                
                for s_item in s_res:
                    q_num = s_item['question_index']
                    gabarito_val = master_dict.get(q_num, "NULO")
                    aluno_val = s_item['choice']
                    
                    is_correct = (gabarito_val == aluno_val) and (aluno_val != "NULO")
                    if is_correct:
                        correct_count += 1
                        
                    table_report.append({
                        "Questão": f"Questão {q_num}",
                        "Gabarito Oficial": gabarito_val,
                        "Resposta do Aluno": aluno_val,
                        "Status": "✅ Acertou" if is_correct else "❌ Errou"
                    })
                    
                    # 🔵 Desenha um retângulo AZUL no número da questão para provar o mapeamento
                    q_b = s_item['q_box']
                    cv2.rectangle(s_img, (int(q_b['x']), int(q_b['y'])), (int(q_b['x']+q_b['w']), int(q_b['y']+q_b['h'])), (255, 0, 0), 2)
                    
                    # 🟢/🔴 Desenha a resposta do aluno
                    c_b = s_item['chosen_box']
                    if c_b:
                        color = (0, 200, 0) if is_correct else (0, 0, 235)
                        cv2.rectangle(s_img, (int(c_b['x']), int(c_b['y'])), (int(c_b['x']+c_b['w']), int(c_b['y']+c_b['h'])), color, 3)
                
                # Exibição dos resultados estruturados
                t1, t2 = st.columns([1.2, 1])
                with t1:
                    st.subheader("🔍 Mapeamento Visual da IA")
                    st.caption("Legenda: Retângulos Azuis = Números das questões detectados | Verdes/Vermelhos = Respostas avaliadas")
                    st.image(cv2.cvtColor(s_img, cv2.COLOR_BGR2RGB), use_container_width=True)
                with t2:
                    st.subheader("📊 Relatório Analítico")
                    total_q = len(s_res)
                    if total_q > 0:
                        st.metric("Nota Final", f"{((correct_count / total_q) * 10):.2f} / 10.0", f"{correct_count} acertos de {total_q} questões")
                    st.dataframe(pd.DataFrame(table_report), use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()
