import streamlit as st
import cv2
import numpy as np
import pandas as pd

# CONFIGURAÇÃO DA INTERFACE
st.set_page_config(page_title="Neural Grid OMR v8.0", page_icon="🧠", layout="wide")

st.markdown("""
    <style>
        .block-container { padding-top: 1.5rem; }
        .stMetric { background-color: #f8f9fa; padding: 12px; border-radius: 8px; border: 1px solid #e9ecef; }
        .report-box { background-color: #ffffff; padding: 20px; border-radius: 8px; border: 1px solid #dee2e6; }
    </style>
""", unsafe_allow_html=True)

class NeuralGridEngine:
    @staticmethod
    def advanced_omr_scan(image_bytes, num_options=5):
        """Processador espacial adaptativo imune a variações de resolução."""
        # 1. Conversão dos bytes para Matriz OpenCV
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return None, None, "Não foi possível decodificar os bytes da imagem."
        
        # 2. Normalização de Escala Base para 1000px de largura
        h, w = img.shape[:2]
        target_w = 1000
        scale = target_w / float(w)
        img_canvas = cv2.resize(img, (target_w, int(h * scale)), interpolation=cv2.INTER_AREA)
        
        # 3. Tratamento de Imagem (Binarização Dinâmica contra sombras)
        gray = cv2.cvtColor(img_canvas, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 51, 9
        )
        
        # 4. Captura de Componentes Visuais
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detected_blobs = []
        
        for cnt in contours:
            x, y, w_b, h_b = cv2.boundingRect(cnt)
            aspect_ratio = w_b / float(h_b)
            
            # Filtro ultra-amplo: aceita qualquer bloco estrutural de tamanho razoável (2% a 25% da tela)
            if 15 <= w_b <= 200 and 15 <= h_b <= 200:
                if 0.5 <= aspect_ratio <= 2.0:
                    area = cv2.contourArea(cnt)
                    # Calcula o preenchimento interno (densidade de pixels pretos)
                    mask = np.zeros(thresh.shape, dtype="uint8")
                    cv2.drawContours(mask, [cnt], -1, 255, -1)
                    density = cv2.mean(thresh, mask=mask)[0]
                    
                    detected_blobs.append({
                        'x': x, 'y': y, 'w': w_b, 'h': h_b,
                        'cx': x + w_b/2.0, 'cy': y + h_b/2.0,
                        'density': density, 'cnt': cnt
                    })
        
        if not detected_blobs:
            return [], img_canvas, "Nenhum bloco de marcação ou questão foi localizado pelo scanner."
            
        # 5. Inteligência de Agrupamento por Linhas (Clusterização Espacial)
        detected_blobs.sort(key=lambda b: b['cy'])
        rows = []
        current_row = [detected_blobs[0]]
        
        # Mediana da altura serve como limite dinâmico de proximidade vertical
        limit_y = np.median([b['h'] for b in detected_blobs]) * 0.7
        
        for b in detected_blobs[1:]:
            mean_y = np.mean([item['cy'] for item in current_row])
            if abs(b['cy'] - mean_y) <= limit_y:
                current_row.append(b)
            else:
                rows.append(current_row)
                current_row = [b]
        rows.append(current_row)
        
        # Filtragem: Mantém apenas linhas que possuem estrutura real de dados (mínimo número + alternativas)
        rows = [r for r in rows if len(r) >= 2]
        rows.sort(key=lambda r: np.mean([item['cy'] for item in r]))
        
        # 6. Mapeamento de Dados (Questão vs Resposta)
        alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
        parsed_data = []
        output_image = img_canvas.copy()
        
        for idx, row in enumerate(rows):
            # Ordena da esquerda para a direita
            row.sort(key=lambda b: b['cx'])
            
            # ÂNCORA: O primeiro elemento à esquerda é obrigatoriamente o número da questão
            question_anchor = row[0]
            alternatives = row[1:]
            
            # Desenha a marcação azul na âncora encontrada
            cv2.rectangle(output_image, (int(question_anchor['x']), int(question_anchor['y'])),
                          (int(question_anchor['x'] + question_anchor['w']), int(question_anchor['y'] + question_anchor['h'])), 
                          (255, 100, 0), 2)
            
            best_option_idx = -1
            max_density = -1
            chosen_box = None
            
            # Varre as alternativas associadas àquela âncora
            for opt_idx, alt in enumerate(alternatives[:num_options]):
                if alt['density'] > max_density and alt['density'] > 60:
                    max_density = alt['density']
                    best_option_idx = opt_idx
                    chosen_box = alt
            
            final_choice = alphabet[best_option_idx] if best_option_idx != -1 else "NULO"
            
            # Se uma resposta foi detectada, desenha um box verde sobre ela
            if chosen_box:
                cv2.rectangle(output_image, (int(chosen_box['x']), int(chosen_box['y'])),
                              (int(chosen_box['x'] + chosen_box['w']), int(chosen_box['y'] + chosen_box['h'])), 
                              (0, 200, 0), 2)
                
            parsed_data.append({
                'question_number': idx + 1,
                'detected_choice': final_choice,
                'box': chosen_box
            })
            
        return parsed_data, output_image, None

# INTERFACE STREAMLIT
def main():
    st.title("🧠 Scanner Inteligente de Gabaritos (Grade Espacial)")
    st.write("Suba as imagens. O sistema renderizará o arquivo imediatamente e aplicará a análise de contornos automaticamente.")
    
    st.sidebar.title("Configurações Gerais")
    num_options = st.sidebar.selectbox("Opções por Linha", [5, 4], format_func=lambda x: f"{x} Alternativas (A até {chr(64+x)})")
    
    app_mode = st.tabs(["📄 Modo Leitura Individual", "📊 Modo Comparação (Gabarito x Aluno)"])
    
    # ----------------------------------------------------
    # TABA 1: LEITURA INDIVIDUAL
    # ----------------------------------------------------
    with app_mode[0]:
        single_file = st.file_uploader("Escolha a imagem do cartão de respostas:", type=['jpg','jpeg','png'], key="single")
        
        if single_file:
            file_bytes = single_file.getvalue()
            
            # GARANTIA: Mostra a imagem original de forma limpa imediatamente
            st.success("✓ Arquivo carregado com sucesso na memória do servidor!")
            
            col_img, col_data = st.columns([1.3, 1])
            
            with col_img:
                st.subheader("🔍 Interpretação do Scanner")
                # Executa a inteligência artificial de leitura
                results, debug_img, err = NeuralGridEngine.advanced_omr_scan(file_bytes, num_options)
                
                if err:
                    st.warning(f"Aviso do Motor: {err}. Exibindo apenas imagem crua.")
                    st.image(file_bytes, use_container_width=True)
                else:
                    st.caption("Legenda Visual: Quadrados Azuis = Números das Questões | Quadrados Verdes = Respostas Computadas")
                    st.image(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB), use_container_width=True)
                    
            with col_data:
                st.subheader("📋 Dados Extraídos")
                if 'results' in locals() and results:
                    df = pd.DataFrame(results)[['question_number', 'detected_choice']]
                    df.columns = ['Número da Questão', 'Alternativa Detectada']
                    st.dataframe(df, use_container_width=True, hide_index=True)
                else:
                    st.info("Aguardando leitura estrutural de componentes.")

    # ----------------------------------------------------
    # TABA 2: COMPARAÇÃO E CORREÇÃO
    # ----------------------------------------------------
    with app_mode[1]:
        c1, c2 = st.columns(2)
        with c1:
            master_file = st.file_uploader("Upload do Gabarito Mestre:", type=['jpg','jpeg','png'], key="m_file")
        with c2:
            student_file = st.file_uploader("Upload da Prova do Aluno:", type=['jpg','jpeg','png'], key="s_file")
            
        if master_file and student_file:
            st.markdown("---")
            if st.button("🚀 Executar Correção Cruzada", use_container_width=True, type="primary"):
                m_res, _, m_err = NeuralGridEngine.advanced_omr_scan(master_file.getvalue(), num_options)
                s_res, s_img, s_err = NeuralGridEngine.advanced_omr_scan(student_file.getvalue(), num_options)
                
                if m_err or s_err:
                    st.error(f"Erro no processamento cruzado: {m_err if m_err else s_err}")
                    return
                
                # Relaciona as respostas do gabarito oficial
                master_map = {item['question_number']: item['detected_choice'] for item in m_res}
                report_list = []
                corrects = 0
                
                for s_item in s_res:
                    q_num = s_item['question_number']
                    gabarito_ans = master_map.get(q_num, "NULO")
                    student_ans = s_item['detected_choice']
                    
                    is_correct = (gabarito_ans == student_ans) and (student_ans != "NULO")
                    if is_correct:
                        corrects += 1
                    
                    report_list.append({
                        "Questão": f"Questão {q_num}",
                        "Gabarito Oficial": gabarito_ans,
                        "Resposta do Aluno": student_ans,
                        "Status": "✅ Correta" if is_correct else "❌ Incorreta"
                    })
                    
                    # Desenha feedback na imagem do aluno (Verde para acerto, Vermelho para erro)
                    box = s_item['box']
                    if box:
                        color = (0, 200, 0) if is_correct else (0, 0, 220)
                        cv2.rectangle(s_img, (int(box['x']), int(box['y'])), 
                                      (int(box['x'] + box['w']), int(box['y'] + box['h'])), color, 3)
                
                # Exibição dos resultados finais
                layout_img, layout_table = st.columns([1.2, 1])
                with layout_img:
                    st.subheader("🔍 Caderno Corrigido do Aluno")
                    st.image(cv2.cvtColor(s_img, cv2.COLOR_BGR2RGB), use_container_width=True)
                with layout_table:
                    st.subheader("📊 Notas e Estatísticas")
                    total_items = len(s_res)
                    if total_items > 0:
                        st.metric("Nota do Aluno", f"{((corrects / total_items) * 10):.2f} / 10.0", f"{corrects} acertos de {total_items}")
                    st.dataframe(pd.DataFrame(report_list), use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()
