import streamlit as st
import cv2
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="Autonomous OMR Intelligence v7.0",
    page_icon="📝",
    layout="wide"
)

st.markdown("""
    <style>
        .block-container { padding-top: 1.5rem; }
        .stMetric { background-color: #f8f9fa; padding: 15px; border-radius: 10px; border: 1px solid #e9ecef; }
    </style>
""", unsafe_allow_html=True)

class AutonomousOMREngine:

    @staticmethod
    def process_single_sheet(image_bytes, num_options=5):
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return None, None, "Erro crítico ao ler os arquivos da imagem."
       
        h, w = img.shape[:2]
        target_w = 900
        scale = target_w / float(w)
        img_resized = cv2.resize(img, (target_w, int(h * scale)), interpolation=cv2.INTER_AREA)
       
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 35, 7
        )
       
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        img_h, img_w = thresh.shape[:2]
       
        min_dim = int(img_w * 0.015)
        max_dim = int(img_w * 0.20)
       
        for cnt in contours:
            x, y, w_box, h_box = cv2.boundingRect(cnt)
            aspect_ratio = w_box / float(h_box)
           
            if min_dim <= w_box <= max_dim and min_dim <= h_box <= max_dim:
                if 0.65 <= aspect_ratio <= 1.4:
                    area = cv2.contourArea(cnt)
                    extent = area / float(w_box * h_box)
                    if extent > 0.38:
                        candidates.append({
                            'x': x, 'y': y, 'w': w_box, 'h': h_box,
                            'cx': x + w_box/2.0, 'cy': y + h_box/2.0, 'cnt': cnt
                        })
                       
        if not candidates:
            return None, img_resized, "A IA não conseguiu isolar blocos estruturais uniformes. Ajuste o enquadramento."
           
        median_w = np.median([c['w'] for c in candidates])
        median_h = np.median([c['h'] for c in candidates])
       
        valid_elements = []
        for c in candidates:
            if (median_w * 0.6) <= c['w'] <= (median_w * 1.5) and (median_h * 0.6) <= c['h'] <= (median_h * 1.5):
                mask = np.zeros(thresh.shape, dtype="uint8")
                cv2.drawContours(mask, [c['cnt']], -1, 255, -1)
                c['density'] = cv2.mean(thresh, mask=mask)[0]
                valid_elements.append(c)
               
        if not valid_elements:
            return None, img_resized, "Os padrões encontrados na imagem estão muito dispersos ou irregulares."
           
        valid_elements.sort(key=lambda e: e['cx'])
        columns_groups = []
        current_col = [valid_elements[0]]
       
        for e in valid_elements[1:]:
            avg_cx = np.mean([item['cx'] for item in current_col])
            if abs(e['cx'] - avg_cx) < (median_w * 0.6):
                current_col.append(e)
            else:
                columns_groups.append(current_col)
                current_col = [e]
        columns_groups.append(current_col)
       
        columns_groups.sort(key=lambda c: np.mean([item['cx'] for item in c]))
        max_col_len = max(len(c) for c in columns_groups)
        valid_columns = [c for c in columns_groups if len(c) >= max(2, max_col_len * 0.4)]
       
        for col_idx, col in enumerate(valid_columns):
            for e in col:
                e['col_idx'] = col_idx
               
        valid_elements.sort(key=lambda e: e['cy'])
        rows_groups = []
        current_row = [valid_elements[0]]
       
        for e in valid_elements[1:]:
            avg_cy = np.mean([item['cy'] for item in current_row])
            if abs(e['cy'] - avg_cy) < (median_h * 0.5):
                current_row.append(e)
            else:
                rows_groups.append(current_row)
                current_row = [e]
        rows_groups.append(current_row)
       
        rows_groups.sort(key=lambda r: np.mean([item['cy'] for item in r]))
       
        options_alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
        total_cols = len(valid_columns)
        start_option_col = max(0, total_cols - num_options)
       
        parsed_sheet = []
        output_img = img_resized.copy()
       
        for row_idx, row in enumerate(rows_groups):
            answer_elements = [e for e in row if 'col_idx' in e and e['col_idx'] >= start_option_col]
            number_elements = [e for e in row if 'col_idx' in e and e['col_idx'] < start_option_col]
           
            if not answer_elements:
                continue
               
            if number_elements:
                for num_e in number_elements:
                    cv2.rectangle(output_img, (int(num_e['x']), int(num_e['y'])),
                                  (int(num_e['x'] + num_e['w']), int(num_e['y'] + num_e['h'])), (255, 120, 0), 2)
           
            highest_density = -1
            chosen_letter = "NULO"
            chosen_box = None
           
            for e in answer_elements:
                letter_pos = e['col_idx'] - start_option_col
                if 0 <= letter_pos < num_options:
                    if e['density'] > highest_density and e['density'] > 75:
                        highest_density = e['density']
                        chosen_letter = options_alphabet[letter_pos]
                        chosen_box = e
           
            if chosen_box:
                cv2.rectangle(output_img, (int(chosen_box['x']), int(chosen_box['y'])),
                              (int(chosen_box['x'] + chosen_box['w']), int(chosen_box['y'] + chosen_box['h'])), (0, 220, 0), 2)
           
            parsed_sheet.append({
                'question_index': row_idx + 1,
                'choice': chosen_letter,
                'box': chosen_box
            })
           
        return parsed_sheet, output_img, None

                        c_b = s_item['box']
                        if c_b:
                            color = (0, 200, 0) if is_correct else (0, 0, 235)
                            cv2.rectangle(s_img, (int(c_b['x']), int(c_b['y'])), (int(c_b['x']+c_b['w']), int(c_b['y']+c_b['h'])), color, 3)
                           
                    t_layout1, t_layout2 = st.columns([1.2, 1])
                    with t_layout1:
                        st.subheader("Correção Visual do Aluno")
                        st.image(cv2.cvtColor(s_img, cv2.COLOR_BGR2RGB), use_container_width=True)
                    with t_layout2:
                        st.subheader("Relatório Comparativo")
                        total_q = len(s_res)
                        if total_q > 0:
                            st.metric("Nota Calculada", f"{((correct_count / total_q) * 10):.2f} / 10.0", f"{correct_count} acertos de {total_q} itens")
                        st.dataframe(pd.DataFrame(table_report), use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()
