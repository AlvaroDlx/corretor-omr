import streamlit as st
import cv2
import numpy as np
import pandas as pd

# Configuração da Página
st.set_page_config(page_title="Corretor OMR v3.0", page_icon="📝", layout="wide")

# ==========================================
# CORE OMNI-PROCESSAMENTO OMR (OpenCV)
# ==========================================
def detect_marked_bubbles(image_bytes, num_options, sens_mode):
    nparr = np.frombuffer(image_bytes, np.uint8)
    src_original = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if src_original is None:
        return [], None

    # 1. NORMALIZAÇÃO DE RESOLUÇÃO (O GRANDE SEGREDO PARA RECONHECIMENTO)
    # Redimensiona qualquer imagem para 1000px de largura mantendo a proporção.
    # Isso impede que fotos de celulares modernos quebrem os filtros matemáticos.
    target_width = 1000
    h_orig, w_orig = src_original.shape[:2]
    scale = target_width / float(w_orig)
    src = cv2.resize(src_original, (target_width, int(h_orig * scale)), interpolation=cv2.INTER_AREA)

    # 2. Pré-processamento de Imagem
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Parâmetros otimizados para a imagem normalizada de 1000px
    block_radius = 35
    subtraction_constant = 10 if sens_mode == 'high' else 5

    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, block_radius, subtraction_constant)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Limites geométricos fixos e calibrados para a escala de 1000px
    min_wh = 12
    max_wh = 55

    bubbles = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)

        # Filtro estrito para formato circular/quadrado da bolinha
        if (min_wh <= w <= max_wh and min_wh <= h <= max_wh and 0.7 <= aspect_ratio <= 1.3):
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

    # 3. CORREÇÃO DE INCLINAÇÃO (DESKEW)
    pts = np.array([[b['cx'], b['cy']] for b in bubbles], dtype=np.float32)
    rect = cv2.minAreaRect(pts)
    angle = rect[-1]
    
    if rect[1][0] < rect[1][1]:
        angle = angle + 90
    if angle > 45: angle -= 90
    elif angle < -45: angle += 90

    if abs(angle) > 0.5:
        M = cv2.getRotationMatrix2D(rect[0], angle, 1.0)
        for b in bubbles:
            b['rcx'] = M[0,0]*b['cx'] + M[0,1]*b['cy'] + M[0,2]
            b['rcy'] = M[1,0]*b['cx'] + M[1,1]*b['cy'] + M[1,2]
    else:
        for b in bubbles:
            b['rcx'] = b['cx']
            b['rcy'] = b['cy']

    # 4. AGRUPAMENTO EM LINHAS (QUESTÕES)
    rows = []
    for bubble in bubbles:
        placed = False
        for row in rows:
            row_center_y = sum(b['rcy'] for b in row) / len(row)
            row_avg_h = sum(b['h'] for b in row) / len(row)
            
            if abs(bubble['rcy'] - row_center_y) < row_avg_h * 0.75:
                row.append(bubble)
                placed = True
                break
        if not placed:
            rows.append([bubble])

    rows.sort(key=lambda r: sum(b['rcy'] for b in r) / len(r))

    options_list = ['A', 'B', 'C', 'D', 'E'][:num_options]
    final_answers = []

    # 5. DETECÇÃO DA RESPOSTA PREENCHIDA
    for index, row in enumerate(rows):
        row.sort(key=lambda b: b['rcx'])

        if len(row) >= min(3, num_options):
            max_density = -1
            marked_index = -1

            for b_idx, bubble in enumerate(row):
                if bubble['density'] > max_density:
                    max_density = bubble['density']
                    marked_index = b_idx

            row_avg_rcy = sum(b['rcy'] for b in row) / len(row)

            # Sensibilidade dinâmica de preenchimento
            cutoff = 25 if sens_mode == 'high' else 40

            if max_density > cutoff and marked_index < num_options:
                final_answers.append({
                    'question': index + 1,
                    'choice': options_list[marked_index],
                    'box': row[marked_index],
                    'rcy': row_avg_rcy
                })
            else:
                final_answers.append({
                    'question': index + 1, 
                    'choice': 'NULO', 
                    'box': row[0],
                    'rcy': row_avg_rcy
                })

    return final_answers, src

# ==========================================
# INTERFACE COM STREAMLIT
# ==========================================
def main():
    st.title("📝 Corretor OMR v3.0")
    st.markdown("Plataforma de correção automatizada com motor de calibração adaptável e redimensionamento inteligente.")

    tab_work, tab_settings, tab_about = st.tabs(["📊 Área de Processamento", "⚙️ Configurações", "ℹ️ Sobre o App"])

    with tab_settings:
        st.header("Configurações do Sistema")
        num_options = st.selectbox("Quantidade de Alternativas por Questão:", [5, 4],
                                   format_func=lambda x: f"{x} Alternativas ({'A, B, C, D, E' if x==5 else 'A, B, C, D'})")
        sens_mode = st.selectbox("Sensibilidade de Detecção da Caneta:", ["normal", "high"],
                                 format_func=lambda x: "Equilibrada (Recomendado)" if x == "normal" else "Alta (Para canetas claras/grafite)")

    with tab_about:
        st.header("Sobre o Projeto")
        st.write("Versão 3.0: Implementado sistema de escala fixa (1000px) que soluciona erros de calibração em fotos de alta resolução.")

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
                with st.spinner("Normalizando imagens e corrigindo..."):
                    
                    master_answers, _ = detect_marked_bubbles(file_master.getvalue(), num_options, sens_mode)
                    student_answers, student_img = detect_marked_bubbles(file_student.getvalue(), num_options, sens_mode)

                    if not master_answers:
                        st.error("⚠️ Erro de Calibração: O sistema não conseguiu detectar os círculos padrões no Gabarito Mestre. Certifique-se de que a imagem contém o cartão de respostas focado e limpo.")
                    else:
                        correct_count = 0
                        results_data = []

                        # Alinhamento Geográfico Proporcional
                        m_min_y = min(q['rcy'] for q in master_answers)
                        m_max_y = max(q['rcy'] for q in master_answers)
                        m_range = m_max_y - m_min_y if m_max_y > m_min_y else 1

                        if student_answers:
                            s_min_y = min(q['rcy'] for q in student_answers)
                            s_max_y = max(q['rcy'] for q in student_answers)
                            s_range = s_max_y - s_min_y if s_max_y > s_min_y else 1
                        else:
                            s_min_y, s_range = 0, 1

                        for m_quest in master_answers:
                            m_rel_y = (m_quest['rcy'] - m_min_y) / m_range
                            
                            best_s_quest = None
                            min_diff = float('inf')
                            
                            if student_answers:
                                for s_quest in student_answers:
                                    s_rel_y = (s_quest['rcy'] - s_min_y) / s_range
                                    diff = abs(m_rel_y - s_rel_y)
                                    if diff < min_diff:
                                        min_diff = diff
                                        best_s_quest = s_quest

                            if best_s_quest is None or min_diff > 0.06:
                                s_quest = {'choice': 'NULO', 'box': None, 'question': m_quest['question']}
                            else:
                                s_quest = best_s_quest

                            is_correct = m_quest['choice'] == s_quest['choice']
                            if is_correct: correct_count += 1
                            
                            results_data.append({
                                "Questão": f"{m_quest['question']}",
                                "Gabarito": m_quest['choice'],
                                "Aluno": s_quest['choice'],
                                "Status": "✅ Correta" if is_correct else "❌ Incorreta"
                            })

                            box = s_quest['box']
                            if box and isinstance(box, dict) and 'x' in box:
                                color = (0, 255, 0) if is_correct else (255, 0, 0)
                                thickness = 3
                                cv2.rectangle(student_img,
                                              (int(box['x'] - 2), int(box['y'] - 2)),
                                              (int(box['x'] + box['w'] + 4), int(box['y'] + box['h'] + 4)),
                                              color, thickness)

                        total = len(master_answers)
                        wrong_count = total - correct_count
                        percent = round((correct_count / total) * 100) if total > 0 else 0
                        grade = f"{(correct_count / total) * 10:.1f}" if total > 0 else "0.0"

                        st.success("🎉 Processamento concluído com sucesso!")
                        
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
                            student_rgb = cv2.cvtColor(student_img, cv2.COLOR_BGR2RGB)
                            st.image(student_rgb, use_container_width=True)
                            
                        with col_tbl:
                            st.subheader("Lista de Questões")
                            df_results = pd.DataFrame(results_data)
                            st.dataframe(df_results, use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()
