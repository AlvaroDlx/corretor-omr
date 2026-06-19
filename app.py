import streamlit as st
import cv2
import numpy as np
import pandas as pd

# Configuração da Página
st.set_page_config(page_title="Corretor OMR v3.5 - Blindado", page_icon="📝", layout="wide")

# ==========================================
# CORE OMNI-PROCESSAMENTO OMR (OpenCV)
# ==========================================
def detect_marked_bubbles(image_bytes, num_options, sens_mode):
    nparr = np.frombuffer(image_bytes, np.uint8)
    src_original = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if src_original is None:
        return [], None, None

    # 1. NORMALIZAÇÃO DE RESOLUÇÃO (Estabiliza o tamanho da imagem)
    target_width = 1000
    h_orig, w_orig = src_original.shape[:2]
    scale = target_width / float(w_orig)
    src = cv2.resize(src_original, (target_width, int(h_orig * scale)), interpolation=cv2.INTER_AREA)

    # 2. PRÉ-PROCESSAMENTO (Filtros de isolamento de tinta)
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Ajuste dinâmico de limiarização para evitar sombras
    block_radius = 31
    subtraction_constant = 8 if sens_mode == 'high' else 4

    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, block_radius, subtraction_constant)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # LIMITES ULTRA-PERMISSIVOS (Para aceitar qualquer câmera/impressão)
    min_wh = 6
    max_wh = 80

    bubbles = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)

        # Filtro geométrico flexível (aceita pequenas distorções de ângulo)
        if (min_wh <= w <= max_wh and min_wh <= h <= max_wh and 0.5 <= aspect_ratio <= 1.8):
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
        return [], src, thresh

    # 3. CORREÇÃO DE INCLINAÇÃO (DESKEW)
    pts = np.array([[b['cx'], b['cy']] for b in bubbles], dtype=np.float32)
    rect = cv2.minAreaRect(pts)
    angle = rect[-1]
    
    if rect[1][0] < rect[1][1]:
        angle = angle + 90
    if angle > 45: angle -= 90
    elif angle < -45: angle += 90

    for b in bubbles:
        if abs(angle) > 0.5:
            M = cv2.getRotationMatrix2D(rect[0], angle, 1.0)
            b['rcx'] = M[0,0]*b['cx'] + M[0,1]*b['cy'] + M[0,2]
            b['rcy'] = M[1,0]*b['cx'] + M[1,1]*b['cy'] + M[1,2]
        else:
            b['rcx'] = b['cx']
            b['rcy'] = b['cy']

    # 4. AGRUPAMENTO EM LINHAS (Tolerância expandida para 85%)
    rows = []
    for bubble in bubbles:
        placed = False
        for row in rows:
            row_center_y = sum(b['rcy'] for b in row) / len(row)
            row_avg_h = sum(b['h'] for b in row) / len(row)
            
            if abs(bubble['rcy'] - row_center_y) < row_avg_h * 0.85:
                row.append(bubble)
                placed = True
                break
        if not placed:
            rows.append([bubble])

    rows.sort(key=lambda r: sum(b['rcy'] for b in r) / len(r))

    options_list = ['A', 'B', 'C', 'D', 'E'][:num_options]
    final_answers = []

    # 5. DETECÇÃO SEM BLOQUEIO (Processa mesmo se a linha estiver incompleta)
    for index, row in enumerate(rows):
        row.sort(key=lambda b: b['rcx'])

        if len(row) > 0:
            max_density = -1
            marked_index = -1

            for b_idx, bubble in enumerate(row):
                if bubble['density'] > max_density:
                    max_density = bubble['density']
                    marked_index = b_idx

            row_avg_rcy = sum(b['rcy'] for b in row) / len(row)
            cutoff = 20 if sens_mode == 'high' else 35

            if max_density > cutoff and marked_index < len(options_list):
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

    return final_answers, src, thresh

# ==========================================
# INTERFACE COM STREAMLIT
# ==========================================
def main():
    st.title("📝 Corretor OMR v3.5 - Versão Blindada")
    st.markdown("Sistema tolerante a falhas físicas com diagnóstico visual em tempo real.")

    tab_work, tab_settings = st.tabs(["📊 Área de Processamento", "⚙️ Configurações Avançadas"])

    with tab_settings:
        st.header("Configurações de Tolerância")
        num_options = st.selectbox("Quantidade de Alternativas por Questão:", [5, 4])
        sens_mode = st.selectbox("Sensibilidade de Captura:", ["normal", "high"],
                                 format_func=lambda x: "Padrão" if x == "normal" else "Alta Eficiência (Para lápis/marcas claras)")

    with tab_work:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("1. Imagem do Gabarito")
            file_master = st.file_uploader("Envie o Gabarito Oficial", type=['png', 'jpg', 'jpeg'], key="master")
            if file_master: st.image(file_master, use_container_width=True)

        with col2:
            st.subheader("2. Imagem da Prova")
            file_student = st.file_uploader("Envie a Prova do Aluno", type=['png', 'jpg', 'jpeg'], key="student")
            if file_student: st.image(file_student, use_container_width=True)

        st.markdown("---")
        
        if file_master and file_student:
            if st.button("🚀 Executar Correção Inteligente", use_container_width=True, type="primary"):
                with st.spinner("Analisando estrutura de pixels..."):
                    
                    master_answers, master_img, master_thresh = detect_marked_bubbles(file_master.getvalue(), num_options, sens_mode)
                    student_answers, student_img, student_thresh = detect_marked_bubbles(file_student.getvalue(), num_options, sens_mode)

                    # Se falhar, o diagnóstico entra em ação imediatamente
                    if not master_answers or not student_answers:
                        st.error("⚠️ Falha Crítica de Leitura: O computador não conseguiu separar os círculos do fundo da imagem.")
                        
                        st.subheader("🔍 Modo Diagnóstico (O que o computador está vendo):")
                        st.warning("Se as caixas abaixo estiverem totalmente pretas ou cheias de manchas brancas puras, aproxime mais a câmera ou melhore a iluminação do papel.")
                        
                        c_diag1, c_diag2 = st.columns(2)
                        with c_diag1:
                            st.write("Visão do Gabarito:")
                            if master_thresh is not None: st.image(master_thresh, caption="Filtro Binário Mestre", use_container_width=True)
                        with c_diag2:
                            st.write("Visão da Prova:")
                            if student_thresh is not None: st.image(student_thresh, caption="Filtro Binário Aluno", use_container_width=True)
                    else:
                        # CORE DA CORREÇÃO GEOGRÁFICA
                        correct_count = 0
                        results_data = []

                        m_min_y = min(q['rcy'] for q in master_answers)
                        m_max_y = max(q['rcy'] for q in master_answers)
                        m_range = m_max_y - m_min_y if m_max_y > m_min_y else 1

                        s_min_y = min(q['rcy'] for q in student_answers)
                        s_max_y = max(q['rcy'] for q in student_answers)
                        s_range = s_max_y - s_min_y if s_max_y > s_min_y else 1

                        for m_quest in master_answers:
                            m_rel_y = (m_quest['rcy'] - m_min_y) / m_range
                            
                            best_s_quest = None
                            min_diff = float('inf')
                            
                            for s_quest in student_answers:
                                s_rel_y = (s_quest['rcy'] - s_min_y) / s_range
                                diff = abs(m_rel_y - s_rel_y)
                                if diff < min_diff:
                                    min_diff = diff
                                    best_s_quest = s_quest

                            # Se houver um abismo de distância física (mais de 8%), a questão foi pulada
                            if best_s_quest is None or min_diff > 0.08:
                                s_quest = {'choice': 'NULO', 'box': None, 'question': m_quest['question']}
                            else:
                                s_quest = best_s_quest

                            is_correct = m_quest['choice'] == s_quest['choice']
                            if is_correct: correct_count += 1
                            
                            results_data.append({
                                "Questão": f"{m_quest['question']}",
                                "Gabarito": m_quest['choice'],
                                "Aluno": s_quest['choice'],
                                "Resultado": "✅ Acertou" if is_correct else "❌ Errou"
                            })

                            box = s_quest['box']
                            if box and isinstance(box, dict) and 'x' in box:
                                color = (0, 255, 0) if is_correct else (255, 0, 0)
                                cv2.rectangle(student_img, (int(box['x']), int(box['y'])), 
                                              (int(box['x'] + box['w']), int(box['y'] + box['h'])), color, 3)

                        total = len(master_answers)
                        st.success(f"🎉 Processado! O aluno acertou {correct_count} de {total} questões.")
                        
                        # Exibição dos dados
                        col_vis, col_tbl = st.columns([1.2, 1])
                        with col_vis:
                            st.subheader("Mapeamento dos Alvos")
                            st.image(cv2.cvtColor(student_img, cv2.COLOR_BGR2RGB), use_container_width=True)
                        with col_tbl:
                            st.subheader("Boletim")
                            st.dataframe(pd.DataFrame(results_data), use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()
