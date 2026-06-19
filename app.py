import streamlit as st
import cv2
import numpy as np
import pandas as pd

# Configuração da Página
st.set_page_config(page_title="Corretor OMR v2.0", page_icon="📝", layout="wide")

# ==========================================
# CORE OMNI-PROCESSAMENTO OMR (OpenCV)
# ==========================================
def detect_marked_bubbles(image_bytes, num_options, sens_mode):
    # Decodifica a imagem recebida em bytes para formato legível pelo OpenCV
    nparr = np.frombuffer(image_bytes, np.uint8)
    src = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if src is None:
        return [], None

    # 1. Normalização do Espaço de Cores e Remoção de Ruído
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # CONFIGURAÇÃO DINÂMICA
    block_radius = 45
    subtraction_constant = 12 if sens_mode == 'high' else 7

    # 2. Limiarização Adaptativa Gaussiana Local
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, block_radius, subtraction_constant)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Limites adaptáveis
    min_wh = max(10, min(src.shape[0], src.shape[1]) * 0.012)
    max_wh = min(src.shape[0], src.shape[1]) * 0.08

    bubbles = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)

        # Filtro Geométrico Restritivo
        if (min_wh <= w <= max_wh and min_wh <= h <= max_wh and 0.65 <= aspect_ratio <= 1.35):
            # Cálculo de preenchimento de pigmentação
            mask = np.zeros(thresh.shape, dtype="uint8")
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            mean = cv2.mean(thresh, mask=mask)
            density = mean[0]

            bubbles.append({
                'x': x, 'y': y, 'w': w, 'h': h,
                'cx': x + w / 2.0,
                'cy': y + h / 2.0,
                'density': density
            })

    if not bubbles:
        return [], src

    # 3. AGRUPAMENTO GEOMÉTRICO (Clustering Dinâmico)
    rows = []
    for bubble in bubbles:
        placed = False
        for row in rows:
            row_center_y = sum(b['cy'] for b in row) / len(row)
            row_avg_h = sum(b['h'] for b in row) / len(row)
            
            if abs(bubble['cy'] - row_center_y) < row_avg_h * 0.65:
                row.append(bubble)
                placed = True
                break
        if not placed:
            rows.append([bubble])

    # Ordena de cima para baixo
    rows.sort(key=lambda r: sum(b['cy'] for b in r) / len(r))

    options_list = ['A', 'B', 'C', 'D', 'E'][:num_options]
    final_answers = []

    # 4. MAPEAMENTO DE DENSIDADE DA QUESTÃO
    for index, row in enumerate(rows):
        # Ordena da esquerda para a direita
        row.sort(key=lambda b: b['x'])

        if len(row) >= min(3, num_options):
            max_density = -1
            marked_index = -1

            for b_idx, bubble in enumerate(row):
                if bubble['density'] > max_density:
                    max_density = bubble['density']
                    marked_index = b_idx

            # Validação de segurança
            if max_density > 35 and marked_index < num_options:
                final_answers.append({
                    'question': index + 1,
                    'choice': options_list[marked_index],
                    'box': row[marked_index]
                })
            else:
                final_answers.append({'question': index + 1, 'choice': 'NULO', 'box': row[0]})

    return final_answers, src

# ==========================================
# INTERFACE COM STREAMLIT
# ==========================================
def main():
    st.title("📝 Corretor OMR v2.0")
    st.markdown("Escaneie, processe e corrija folhas de respostas instantaneamente.")

    # Menu de Navegação usando Tabs
    tab_work, tab_settings, tab_about = st.tabs(["📊 Área de Processamento", "⚙️ Configurações", "ℹ️ Sobre o App"])

    with tab_settings:
        st.header("Configurações do Sistema")
        num_options = st.selectbox("Quantidade de Alternativas por Questão:", [5, 4],
                                   format_func=lambda x: f"{x} Alternativas ({'A, B, C, D, E' if x==5 else 'A, B, C, D'})")
        sens_mode = st.selectbox("Sensibilidade de Detecção da Caneta:", ["normal", "high"],
                                 format_func=lambda x: "Equilibrada (Recomendado)" if x == "normal" else "Alta (Para canetas claras/grafite)")

    with tab_about:
        st.header("Sobre o Projeto")
        st.write("Este utilitário realiza processamento de imagem baseado em visão computacional 100% local (Python), sem enviar seus dados para serviços em nuvem de terceiros.")
        st.markdown("""
        **Dicas para captura perfeita:**
        * Mantenha a folha em uma superfície plana e bem iluminada.
        * Evite sombras severas cruzando as bolinhas de marcação.
        * Evite inclinações extremas da câmera; fotografe o mais paralelo possível à folha.
        """)

    with tab_work:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("1. Gabarito Mestre")
            file_master = st.file_uploader("Envie a imagem do Gabarito Oficial", type=['png', 'jpg', 'jpeg'], key="master")
            if file_master:
                st.image(file_master, use_container_width=True)

        with col2:
            st.subheader("2. Prova do Aluno")
            file_student = st.file_uploader("Envie a imagem da Prova do Aluno", type=['png', 'jpg', 'jpeg'], key="student")
            if file_student:
                st.image(file_student, use_container_width=True)

        st.markdown("---")
        
        if file_master and file_student:
            if st.button("✅ Corrigir Gabarito Agora", use_container_width=True, type="primary"):
                with st.spinner("Analisando matriz de pixels..."):
                    
                    master_answers, _ = detect_marked_bubbles(file_master.getvalue(), num_options, sens_mode)
                    student_answers, student_img = detect_marked_bubbles(file_student.getvalue(), num_options, sens_mode)

                    if not master_answers:
                        st.error("⚠️ Erro de Calibração: O motor OMR não conseguiu localizar os círculos do Gabarito Mestre. Verifique a iluminação ou mude as Configurações de Sensibilidade.")
                    else:
                        # Processamento dos Resultados
                        correct_count = 0
                        results_data = []

                        for index, m_quest in enumerate(master_answers):
                            s_quest = student_answers[index] if index < len(student_answers) else {'choice': 'NULO', 'box': None}
                            is_correct = m_quest['choice'] == s_quest['choice']
                            
                            if is_correct: correct_count += 1
                            
                            results_data.append({
                                "Questão": f"{m_quest['question']}",
                                "Gabarito": m_quest['choice'],
                                "Aluno": s_quest['choice'],
                                "Status": "✅ Correta" if is_correct else "❌ Incorreta"
                            })

                            # Desenho do Feedback Visual
                            box = s_quest['box']
                            if box and isinstance(box, dict) and 'x' in box:
                                color = (0, 255, 0) if is_correct else (255, 0, 0) # Cores corrigidas para RGB no Streamlit
                                thickness = max(3, int(student_img.shape[1] * 0.004))
                                cv2.rectangle(student_img,
                                              (int(box['x'] - 3), int(box['y'] - 3)),
                                              (int(box['x'] + box['w'] + 6), int(box['y'] + box['h'] + 6)),
                                              color, thickness)

                        total = len(master_answers)
                        wrong_count = total - correct_count
                        percent = round((correct_count / total) * 100) if total > 0 else 0
                        grade = f"{(correct_count / total) * 10:.1f}" if total > 0 else "0.0"

                        st.success("🎉 Correção finalizada com sucesso!")
                        
                        # Painel de Métricas
                        st.header("📊 Painel de Resultados")
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Nota Final", grade)
                        m2.metric("Acertos", correct_count)
                        m3.metric("Erros", wrong_count)
                        m4.metric("Aproveitamento", f"{percent}%")

                        st.markdown("---")
                        
                        col_vis, col_tbl = st.columns([1.2, 1])
                        with col_vis:
                            st.subheader("Mapeamento Computacional")
                            # Converte BGR para RGB para o Streamlit exibir com cores certas
                            student_rgb = cv2.cvtColor(student_img, cv2.COLOR_BGR2RGB)
                            st.image(student_rgb, use_container_width=True)
                            
                        with col_tbl:
                            st.subheader("Lista de Questões")
                            df_results = pd.DataFrame(results_data)
                            st.dataframe(df_results, use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()
