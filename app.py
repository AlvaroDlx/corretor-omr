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
    page_icon="📝",
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
        h_img = max(600, 150 + num_questions * 55)
        img = np.ones((h_img, 1000, 3), dtype=np.uint8) * 255
       
        cv2.putText(img, f"GABARITO SINTETICO SIMULADO: {pattern_type.upper()}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 80), 2)
       
        for q in range(num_questions):
            y = 120 + q * 55
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
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        morphed = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel)
       
        block_size = 55
        constant = 12 if sensitivity_mode == 'high' else 8
        return cv2.adaptiveThreshold(morphed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, constant)

    @classmethod
    def extract_and_filter_contours(cls, thresh_img):
        contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        potential_bubbles = []
       
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h) if h > 0 else 0
            area = cv2.contourArea(cnt)
           
            if h > 0 and 0.5 <= aspect_ratio <= 1.5:
                extent = area / float(w * h)
                if extent > 0.3:
                    potential_bubbles.append({'x': x, 'y': y, 'w': w, 'h': h, 'cnt': cnt, 'cx': x + w/2.0, 'cy': y + h/2.0})

        if not potential_bubbles:
            return []

        median_w = np.median([b['w'] for b in potential_bubbles])
        valid_bubbles = []
       
        for b in potential_bubbles:
            if median_w * 0.5 <= b['w'] <= median_w * 1.5:
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

        # CORREÇÃO DO AGRUPAMENTO: Agrupamento robusto por tolerância de faixa do centro geométrico (cy)
        bubbles.sort(key=lambda b: b['cy'])
        rows = []
        while len(bubbles) > 0:
            base_cy = bubbles[0]['cy']
            # Agrupa tudo que pertence à mesma linha horizontal da primeira bolha atual
            row = [b for b in bubbles if abs(b['cy'] - base_cy) < 25]
            # Ordena os elementos da linha estritamente da esquerda para a direita
            row.sort(key=lambda b: b['cx'])
            rows.append(row)
            # Remove as bolhas processadas
            bubbles = [b for b in bubbles if b not in row]
       
        # Ordena as linhas completas de cima para baixo
        rows.sort(key=lambda r: r[0]['cy'])
       
        options_alphabet = ['A', 'B', 'C', 'D', 'E'][:num_options]
        parsed_results = []
        target_rows = rows[:num_questions]
       
        for q_index, row in enumerate(target_rows):
            # Garante que só usaremos o número de opções configuradas a partir da direita/esquerda corretamente
            if len(row) > num_options:
                row = row[:num_options]
               
            best_bubble_idx = -1
            max_density = -1
            baseline_density = 15 if sensitivity_mode == 'high' else 30
           
            for idx, b in enumerate(row):
                if b['density'] > max_density:
                    max_density = b['density']
                    best_bubble_idx = idx

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
    st.sidebar.title("Configurações do Motor")
   
    num_questions = st.sidebar.slider("Número de Questões", min_value=5, max_value=30, value=10, step=1, key="slider_q")
    num_options = st.sidebar.selectbox("Opções por Questão", [5, 4], format_func=lambda x: f"{x} Alternativas (A a {chr(64+x)})", key="select_opt")
    sens_mode = st.sidebar.select_slider("Sensibilidade do Scanner", options=['normal', 'high'], format_func=lambda x: "Padrão (Caneta)" if x=='normal' else "Alta (Grafite/Lápis)", key="slider_sens")
   
    st.sidebar.markdown("---")
    st.sidebar.caption("Enterprise OMR Engine v5.0")

    st.title("📝 Dashboard de Correção Inteligente OMR")

    tabs = st.tabs(["📊 Painel de Operações", "🔬 Diagnóstico da IA e Filtros"])

    with tabs[0]:
        c1, c2 = st.columns(2)
       
        with c1:
            st.subheader("📋 Gabarito de Calibração (Mestre)")
            master_file = st.file_uploader("Upload da folha gabarito", type=['jpg','jpeg','png'], key="m_upl")
            if master_file:
                m_bytes = master_file.getvalue()
                st.image(master_file, caption="Gabarito de Referência", use_container_width=True)
            else:
                m_bytes = OMRProcessingEngine.generate_demo_sheet('master', num_questions, num_options)
                st.image(m_bytes, caption="[MODO EMBUTIDO] Gabarito Mestre", use_container_width=True)
           
        with c2:
            st.subheader("📝 Cartão de Respostas (Alunos)")
            student_file = st.file_uploader("Upload da folha do aluno", type=['jpg','jpeg','png'], key="s_upl")
            if student_file:
                s_bytes = student_file.getvalue()
                st.image(student_file, caption="Cartão do Aluno

