import streamlit as st
import cv2
import numpy as np
import pandas as pd
import io

# ==========================================
# CONFIGURAÇÃO DE INTERFACE DA PLATAFORMA
# ==========================================
st.set_page_config(
    page_title="Enterprise OMR Analytics v5.0", 
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
# MÓDULO DE INTELIGÊNCIA GEOMÉTRICA (OMR)
# ==========================================
class OMRProcessingEngine:
    
    @staticmethod
    def generate_demo_sheet(pattern_type, num_questions, num_options):
        h_img = max(600, 100 + num_questions * 55)
        img = np.ones((h_img, 1000, 3), dtype=np.uint8) * 255
        
        cv2.putText(img, f"GABARITO SINTETICO SIMULADO: {pattern_type.upper()}", (50, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 80), 2)
        
        for q in range(num_questions):
            y = 100 + q * 50
            cv2.putText(img, f"Q{q+1:02d}:", (50, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2)
            
            if pattern_type == 'master':
                marked_option = q % num_options
            else:
                marked_option = 0 if q % 3 == 0 else q % num_options
                    
            for opt in range(num_options):
                x = 220 + opt * 130
                cv2.circle(img, (x, y), 20, (0, 0, 0), 2)
                letra = chr(65 + opt)
                cv2.putText(img, letra, (x - 7, y + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 120, 120), 2)
                
                if opt == marked_option:
                    cv2.circle(img, (x, y), 18, (40, 40, 40), -1)
                    
        _, buffer = cv2.imencode('.png', img)
        return buffer.tobytes()

    @staticmethod
    def normalize_resolution(image_bytes, target_width=1000):
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Formato de imagem inválido.")
        h, w = img.shape[:2]
        scale = target_width / float(w)
        return cv2.resize(img, (target_width, int(h * scale)), interpolation=cv2.INTER_AREA), scale

    @staticmethod
    def apply_computer_vision_filters(gray_img, sensitivity_mode):
        blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)
        block_size = 55 # Aumentado para ignorar sombras ainda maiores
        constant = 12 if sensitivity_mode == 'high' else 8
        return cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, constant)

    @classmethod
    def extract_and_filter_contours(cls, thresh_img):
        """Inteligência Morfológica Dinâmica."""
        contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        potential_bubbles = []
        
        # PASSO 1: Encontrar tudo que parece um círculo sólido
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h)
            area = cv2.contourArea(cnt)
            
            if h > 0 and 0.7 <= aspect_ratio <= 1.3:
                extent = area / float(w * h)
                # Filtro de solidez: ignora letras "finas"
                if extent > 0.4:
                    potential_bubbles.append({'x': x, 'y': y, 'w': w, 'h': h, 'cnt': cnt, 'cx': x + w/2.0, 'cy': y + h/2.0})

        if not potential_bubbles: return []

        # PASSO 2: Calibração Dinâmica (O Segredo)
        # Calcula a largura média das bolhas. Isso permite analisar fotos de perto ou de longe automaticamente.
        median_w = np.median([b['w'] for b in potential_bubbles])
        
        valid_bubbles = []
        for b in potential_bubbles:
            # Pega apenas o que tiver um tamanho parecido com a média (ignora caixas grandes e ruídos pequenos)
            if median_w * 0.6 <= b['w'] <= median_w * 1.4:
                mask = np.zeros(thresh_img.shape, dtype="uint8")
                cv2.drawContours(mask, [b['cnt']], -1, 255, -1)
                density = cv2.mean(thresh_img, mask=mask)[0]
                b['density'] = density
                valid_bubbles.append(b)
                
        return valid_bubbles

    @classmethod
    def process_form(cls, image_bytes, num_questions, num_options, sensitivity_mode):
        src, scale = cls.normalize_resolution(image_bytes)
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        thresh = cls.apply_computer_vision_filters(gray, sensitivity_mode)
        bubbles = cls.extract_and_filter_contours(thresh)
        
        if not bubbles:
            return None, src, thresh, "Erro Crítico: Nenhuma marcação ou bolha identificada."

        # PENSAMENTO ESTRUTURAL GEOGRÁFICO (Substitui o K-Means)
        # Ordena tudo de cima para baixo
        bubbles.sort(key=lambda b: b['cy'])
        
        rows = []
        current_row = [bubbles[0]]
        median_h = np.median([b['h'] for b in bubbles])

        # Agrupa bolhas em linhas baseadas na distância vertical
        for b in bubbles[1:]:
            if abs(b['cy'] - current_row[-1]['cy']) < (median_h * 0.8):
                current_row.append(b)
            else:
                rows.append(current_row)
                current_row = [b]
        rows.append(current_row)
        
        options_alphabet = ['A', 'B', 'C', 'D', 'E'][:num_options]
        parsed_results = []
        
        # Pega apenas a quantidade de linhas que o usuário configurou (de cima para baixo)
        target_rows = rows[:num_questions]
        
        for q_index, row in enumerate(target_rows):
            # Ordena da Esquerda para a Direita
            row.sort(key=lambda b: b['cx'])
            
            # Corta ruídos lidos à esquerda (como números da questão)
            if len(row) > num_options:
                row = row[-num_options:]
                
            # LÓGICA RELATIVA DE PREENCHIMENTO
            # Em vez de um limite fixo, compara qual bolha é a mais escura da própria linha
            best_bubble_idx = -1
            max_density = -1
            baseline_density = 15 if sensitivity_mode == 'high' else 25 # Mínimo absoluto para não chutar em branco
            
            for idx, b in enumerate(row):
                if b['density'] > max_density:
                    max_density = b['density']
                    best_bubble_idx = idx

            # Registra a decisão da IA
            if max_density > baseline_density and best_bubble_idx < len(row):
                safe_index = min(best_bubble_idx, num_options - 1)
                final_choice = options_alphabet[safe_index]
                chosen_box = row[safe_index]
                parsed_results.append({'question': q_index + 1, 'choice': final_choice, 'box': chosen_box})
            else:
                parsed_results.append({'question': q_index + 1, 'choice': 'NULO', 'box': row[0] if row else None})
                
        return parsed_results, src, thresh, None

# ==========================================
# INTERFACE GRÁFICA DE USUÁRIO (STREAMLIT UI)
# ==========================================
def main():
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=80)
    st.sidebar.title("Configurações do Motor")
    
    num_questions = st.sidebar.slider("Número de Questões", min_value=5, max_value=30, value=10, step=1)
    num_options = st.sidebar.selectbox("Opções por Questão", [5, 4], format_func=lambda x: f"{x} Alternativas (A a {chr(64+x)})")
    sens_mode = st.sidebar.select_slider("Sensibilidade do Scanner", options=['normal', 'high'], format_func=lambda x: "Padrão (Caneta)" if x=='normal' else "Alta (Grafite/Lápis)")
    
    st.sidebar.markdown("---")
    st.sidebar.caption("Enterprise OMR Engine v5.0")

    st.title("🧠 Dashboard de Correção Inteligente OMR")

    tabs = st.tabs(["📊 Painel de Operações", "🔍 Diagnóstico da IA e Filtros"])

    with tabs[0]:
        c1, c2 = st.columns(2)
        
        with c1:
            st.subheader("📁 Gabarito de Calibração (Mestre)")
            master_file = st.file_uploader("Upload da folha gabarito", type=['jpg','jpeg','png'], key="m_upl")
            if master_file:
                m_bytes = master_file.getvalue()
                st.image(master_file, caption="Gabarito de Referência", use_container_width=True)
            else:
                m_bytes = OMRProcessingEngine.generate_demo_sheet('master', num_questions, num_options)
                st.image(m_bytes, caption="📸 [MODO EMBUTIDO] Gabarito Mestre", use_container_width=True)
            
        with c2:
            st.subheader("📁 Cartão de Respostas (Alunos)")
            student_file = st.file_uploader("Upload da folha do aluno", type=['jpg','jpeg','png'], key="s_upl")
            if student_file:
                s_bytes = student_file.getvalue()
                st.image(student_file, caption="Cartão do Aluno", use_container_width=True)
            else:
                s_bytes = OMRProcessingEngine.generate_demo_sheet('student', num_questions, num_options)
                st.image(s_bytes, caption="📸 [MODO EMBUTIDO] Respostas do Aluno", use_container_width=True)

        st.markdown("---")
        if st.button("🚀 Iniciar Análise e Correção em Lote", use_container_width=True, type="primary"):
            
            with st.spinner("Analisando..."):
                m_res, m_img, m_thresh, m_err = OMRProcessingEngine.process_form(m_bytes, num_questions, num_options, sens_mode)
                s_res, s_img, s_thresh, s_err = OMRProcessingEngine.process_form(s_bytes, num_questions, num_options, sens_mode)
                
                st.session_state['m_thresh'] = m_thresh
                st.session_state['s_thresh'] = s_thresh

                if m_err or s_err:
                    st.error(f"❌ Erro Estrutural: {m_err if m_err else s_err}")
                else:
                    correct_answers = 0
                    report_data = []
                    
                    for m_data, s_data in zip(m_res, s_res):
                        match = (m_data['choice'] == s_data['choice']) and (m_data['choice'] != 'NULO')
                        if match: correct_answers += 1
                        
                        report_data.append({
                            "Questão ID": f"Questão {m_data['question']}",
                            "Gabarito Oficial": m_data['choice'],
                            "Resposta Aluno": s_data['choice'],
                            "Validação": "✅ Correta" if match else "❌ Incorreta"
                        })
                        
                        box = s_data['box']
                        if box:
                            draw_color = (0, 200, 0) if match else (0, 0, 230)
                            cv2.rectangle(s_img, (int(box['x']), int(box['y'])), 
                                          (int(box['x'] + box['w']), int(box['y'] + box['h'])), draw_color, 3)
                    
                    st.subheader("📊 Relatório de Performance Estatística")
                    idx1, idx2, idx3, idx4 = st.columns(4)
                    
                    final_grade = (correct_answers / num_questions) * 10
                    idx1.metric("Nota do Aluno", f"{final_grade:.2f} / 10.0")
                    idx2.metric("Total de Acertos", f"{correct_answers} itens")
                    idx3.metric("Erros/Nulos", f"{num_questions - correct_answers} itens")
                    idx4.metric("Aproveitamento Bruto", f"{int((correct_answers/num_questions)*100)}%")
                    
                    st.markdown("---")
                    
                    layout_left, layout_right = st.columns([1.3, 1])
                    
                    with layout_left:
                        st.subheader("🔍 Validação Visográfica da IA")
                        st.image(cv2.cvtColor(s_img, cv2.COLOR_BGR2RGB), use_container_width=True)
                        
                    with layout_right:
                        st.subheader("📋 Matriz de Respostas Computadas")
                        df_report = pd.DataFrame(report_data)
                        st.dataframe(df_report, use_container_width=True, hide_index=True)

    with tabs[1]:
        st.header("Análise Avançada do Espectro Binário")
        c_diag1, c_diag2 = st.columns(2)
        with c_diag1:
            st.subheader("Matriz Binária - Gabarito Mestre")
            if 'm_thresh' in st.session_state:
                st.image(st.session_state['m_thresh'], use_container_width=True, channels="GRAY")
        with c_diag2:
            st.subheader("Matriz Binária - Prova Aluno")
            if 's_thresh' in st.session_state:
                st.image(st.session_state['s_thresh'], use_container_width=True, channels="GRAY")

if __name__ == "__main__":
    main()
