import streamlit as st
import cv2
import numpy as np
import pandas as pd
import io
from sklearn.cluster import KMeans

# ==========================================
# CONFIGURAÇÃO DE INTERFACE DA PLATAFORMA
# ==========================================
st.set_page_config(
    page_title="Enterprise OMR Analytics v4.5", 
    page_icon="🧠", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilização customizada para deixar a interface profissional
st.markdown("""
    <style>
        .block-container { padding-top: 2rem; }
        .stMetric { background-color: #f8f9fa; padding: 15px; border-radius: 10px; border: 1px solid #e9ecef; }
        .stAlert { border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# MÓDULO MATEMÁTICO E ENGENHARIA DE SINAIS (OMR)
# ==========================================
class OMRProcessingEngine:
    """Classe responsável pelo ciclo de vida do processamento de imagens e IA."""
    
    @staticmethod
    def normalize_resolution(image_bytes, target_width=1000):
        """Redimensiona a imagem dinamicamente mantendo o aspect ratio original."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Formato de imagem inválido ou corrompido.")
        
        h, w = img.shape[:2]
        scale = target_width / float(w)
        resized_img = cv2.resize(img, (target_width, int(h * scale)), interpolation=cv2.INTER_AREA)
        return resized_img, scale

    @staticmethod
    def apply_computer_vision_filters(gray_img, sensitivity_mode):
        """Aplica mascaramento adaptativo e remoção de ruídos de alta frequência."""
        blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)
        block_size = 35
        constant = 8 if sensitivity_mode == 'high' else 4
        
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, block_size, constant
        )
        return thresh

    @classmethod
    def extract_and_filter_contours(cls, thresh_img):
        """Varre a matriz binária isolando contornos com morfologia circular."""
        contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_bubbles = []
        
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h)
            
            # Filtros anatômicos rígidos baseados na resolução de 1000px
            if (12 <= w <= 65 and 12 <= h <= 65 and 0.65 <= aspect_ratio <= 1.35):
                mask = np.zeros(thresh_img.shape, dtype="uint8")
                cv2.drawContours(mask, [cnt], -1, 255, -1)
                density = cv2.mean(thresh_img, mask=mask)[0]
                
                valid_bubbles.append({
                    'x': x, 'y': y, 'w': w, 'h': h,
                    'cx': x + w / 2.0, 'cy': y + h / 2.0,
                    'density': density
                })
        return valid_bubbles

    @classmethod
    def process_form(cls, image_bytes, num_questions, num_options, sensitivity_mode):
        """Orquestrador do processamento de imagens integrado com Machine Learning."""
        # 1. Pipeline de visão computacional clássica
        src, scale = cls.normalize_resolution(image_bytes)
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        thresh = cls.apply_computer_vision_filters(gray, sensitivity_mode)
        bubbles = cls.extract_and_filter_contours(thresh)
        
        if len(bubbles) < num_questions:
            return None, src, thresh, f"Déficit de Captura: Apenas {len(bubbles)} círculos identificados. Mínimo esperado: {num_questions}."
        
        # 2. Motor de IA - Clusterização Espacial K-Means
        y_coords = np.array([[b['cy']] for b in bubbles])
        kmeans = KMeans(n_clusters=num_questions, random_state=42, n_init='auto').fit(y_coords)
        
        # Agrupamento das bolinhas por linhas (questões)
        rows_map = {i: [] for i in range(num_questions)}
        for bubble, label in zip(bubbles, kmeans.labels_):
            rows_map[label].append(bubble)
            
        # Ordenação cronológica de linhas (Top -> Bottom)
        sorted_labels = sorted(rows_map.keys(), key=lambda k: np.mean([b['cy'] for b in rows_map[k]]))
        
        options_alphabet = ['A', 'B', 'C', 'D', 'E'][:num_options]
        parsed_results = []
        
        # 3. Decodificação de preenchimento por densidade de grafite/tinta
        for q_index, label in enumerate(sorted_labels):
            current_row = rows_map[label]
            current_row.sort(key=lambda b: b['cx']) # Esquerda para a Direita
            
            max_density = -1
            detected_option_index = -1
            cutoff_threshold = 22 if sensitivity_mode == 'high' else 38
            
            for b_idx, b in enumerate(current_row):
                if b['density'] > max_density:
                    max_density = b['density']
                    detected_option_index = b_idx
            
            if max_density > cutoff_threshold and detected_option_index < len(current_row):
                safe_index = min(detected_option_index, num_options - 1)
                
                if len(current_row) >= num_options:
                    final_choice = options_alphabet[detected_option_index] if detected_option_index < num_options else 'NULO'
                    chosen_box = current_row[detected_option_index] if detected_option_index < num_options else current_row[0]
                else:
                    final_choice = options_alphabet[safe_index]
                    chosen_box = current_row[safe_index]
                    
                parsed_results.append({'question': q_index + 1, 'choice': final_choice, 'box': chosen_box})
            else:
                parsed_results.append({'question': q_index + 1, 'choice': 'NULO', 'box': current_row[0] if current_row else None})
                
        return parsed_results, src, thresh, None

# ==========================================
# INTERFACE GRÁFICA DE USUÁRIO (STREAMLIT UI)
# ==========================================
def main():
    # Sidebar de Controle Avançado
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=80)
    st.sidebar.title("Configurações do Motor")
    st.sidebar.markdown("Ajuste as propriedades estruturais da prova antes de rodar o algoritmo de IA.")
    
    num_questions = st.sidebar.slider("Número de Questões", min_value=5, max_value=100, value=10, step=1)
    num_options = st.sidebar.selectbox("Opções por Questão", [5, 4], format_func=lambda x: f"{x} Alternativas (A a {chr(64+x)})")
    sens_mode = st.sidebar.select_slider("Sensibilidade do Scanner", options=['normal', 'high'], format_func=lambda x: "Padrão (Caneta)" if x=='normal' else "Alta (Grafite/Lápis)")
    
    st.sidebar.markdown("---")
    st.sidebar.caption("Enterprise OMR Engine v4.5 • Powered by Scikit-Learn & OpenCV")

    # Cabeçalho Principal
    st.title("🧠 Dashboard de Correção Inteligente OMR")
    st.markdown("Plataforma integrada de Visão Computacional e Machine Learning para processamento em larga escala de cartões-resposta.")

    tabs = st.tabs(["📊 Painel de Operações", "🔍 Diagnóstico da IA e Filtros", "📋 Documentação Técnica"])

    with tabs[2]:
        st.header("Documentação do Sistema")
        st.markdown("""
        ### Funcionamento do Pipeline de IA:
        1. **Normalização Espacial:** Imagens recebidas são redimensionadas em uma matriz fixa de 1000 pixels horizontais para mitigar variações de hardware de câmeras.
        2. **Filtragem por Distribuição Gaussiana Adaptativa:** Remove sombras dinâmicas e realça variações de contraste causadas por canetas esferográficas.
        3. **Clusterização por K-Means:** O algoritmo divide as coordenadas geográficas $Y$ em agrupamentos homogêneos baseados no hiperparâmetro de questões inserido.
        """)

    with tabs[0]:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("📁 Gabarito de Calibração (Mestre)")
            master_file = st.file_uploader("Upload da folha gabarito", type=['jpg','jpeg','png'], key="m_upl")
            if master_file: st.image(master_file, caption="Gabarito de Referência carregado.", use_container_width=True)
            
        with c2:
            st.subheader("📁 Cartão de Respostas (Aluno)")
            student_file = st.file_uploader("Upload da folha do aluno", type=['jpg','jpeg','png'], key="s_upl")
            if student_file: st.image(student_file, caption="Cartão do Aluno carregado.", use_container_width=True)

        if master_file and student_file:
            st.markdown("---")
            if st.button("🚀 Iniciar Análise e Correção em Lote", use_container_width=True, type="primary"):
                
                with st.spinner("Executando segmentação e classificação de pixels..."):
                    m_res, m_img, m_thresh, m_err = OMRProcessingEngine.process_form(master_file.getvalue(), num_questions, num_options, sens_mode)
                    s_res, s_img, s_thresh, s_err = OMRProcessingEngine.process_form(student_file.getvalue(), num_questions, num_options, sens_mode)
                    
                    # Armazena em cache os filtros binários para visualização na Tab de diagnósticos
                    st.session_state['m_thresh'] = m_thresh
                    st.session_state['s_thresh'] = s_thresh

                    if m_err or s_err:
                        st.error(f"❌ Abortado por Erro Estrutural: {m_err if m_err else s_err}")
                        st.info("Acesse a aba 'Diagnóstico da IA e Filtros' para inspecionar visualmente a leitura da câmera.")
                    else:
                        # Geração do Relatório de Desempenho
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
                            
                            # Renderização dos Bounding Boxes de Feedback Visual
                            box = s_data['box']
                            if box:
                                # Em OpenCV as cores são BGR: (0, 200, 0) é verde, (0, 0, 230) é vermelho
                                draw_color = (0, 200, 0) if match else (0, 0, 230)
                                cv2.rectangle(s_img, (int(box['x']), int(box['y'])), 
                                              (int(box['x'] + box['w']), int(box['y'] + box['h'])), draw_color, 3)
                        
                        # Bloco de Métricas de Alta Performance
                        st.subheader("📊 Relatório de Performance Estatística")
                        idx1, idx2, idx3, idx4 = st.columns(4)
                        
                        final_grade = (correct_answers / num_questions) * 10
                        idx1.metric("Nota do Aluno", f"{final_grade:.2f} / 10.0")
                        idx2.metric("Total de Acertos", f"{correct_answers} itens")
                        idx3.metric("Erros/Nulos", f"{num_questions - correct_answers} itens")
                        idx4.metric("Aproveitamento Bruto", f"{int((correct_answers/num_questions)*100)}%")
                        
                        st.markdown("---")
                        
                        # Layout de Exibição Dupla (Imagem Processada vs Tabela)
                        layout_left, layout_right = st.columns([1.3, 1])
                        
                        with layout_left:
                            st.subheader("🔍 Validação Visográfica da IA")
                            st.image(cv2.cvtColor(s_img, cv2.COLOR_BGR2RGB), use_container_width=True, caption="Quadrantes verdes indicam acerto. Vermelhos indicam divergência.")
                            
                        with layout_right:
                            st.subheader("📋 Matriz de Respostas Computadas")
                            df_report = pd.DataFrame(report_data)
                            st.dataframe(df_report, use_container_width=True, hide_index=True)
                            
                            # Funcionalidade Enterprise: Exportação para CSV/Excel
                            csv_buffer = io.StringIO()
                            df_report.to_csv(csv_buffer, index=False)
                            st.download_button(
                                label="📥 Exportar Dados para Excel/CSV",
                                data=csv_buffer.getvalue(),
                                file_name="relatorio_correcao_omr.csv",
                                mime="text/csv",
                                use_container_width=True
                            )

    with tabs[1]:
        st.header("Análise Avançada do Espectro Binário")
        st.markdown("Esta seção serve para auditar o comportamento dos algoritmos e verificar interferências de luz ou artefatos no papel.")
        
        c_diag1, c_diag2 = st.columns(2)
        with c_diag1:
            st.subheader("Matriz Binária - Gabarito Mestre")
            if 'm_thresh' in st.session_state:
                st.image(st.session_state['m_thresh'], use_container_width=True, channels="GRAY")
            else:
                st.info("Execute o processamento para gerar o mapa de espectro.")
        with c_diag2:
            st.subheader("Matriz Binária - Prova Aluno")
            if 's_thresh' in st.session_state:
                st.image(st.session_state['s_thresh'], use_container_width=True, channels="GRAY")
            else:
                st.info("Execute o processamento para gerar o mapa de espectro.")

if __name__ == "__main__":
    main()
