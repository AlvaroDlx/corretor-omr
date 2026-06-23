import streamlit as st
import cv2
import numpy as np
import pandas as pd

# ==========================================
# CONFIGURAÇÃO DO SITE (NÃO MEXER)
# ==========================================
st.set_page_config(page_title="Leitor OMR Estabilizado", layout="wide")

st.title("🧠 Leitor de Gabaritos Oficial")
st.write("Faça o upload da imagem. O sistema irá carregar o arquivo na tela e mapear as questões e respostas.")

# ==========================================
# INTERFACE DE UPLOAD
# ==========================================
uploaded_file = st.file_uploader("Selecione a foto do gabarito (Qualquer resolução):", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # 1. GARANTIA DE CARREGAMENTO: Lê os bytes e transforma em imagem
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if img is None:
        st.error("Erro ao carregar o arquivo de imagem. Tente outro formato.")
    else:
        # 2. PADRONIZAÇÃO DE ESCALA (Trabalha sempre na mesma proporção)
        h, w = img.shape[:2]
        largura_padrao = 1000
        proporcao = largura_padrao / float(w)
        img_redimensionada = cv2.resize(img, (largura_padrao, int(h * proporcao)), interpolation=cv2.INTER_AREA)
        
        # Exibe a imagem pura direto no site para confirmar o upload
        st.success("✓ Imagem carregada com sucesso!")
        
        # 3. PROCESSAMENTO DA IA PARA ENCONTRAR OS ELEMENTOS
        gray = cv2.cvtColor(img_redimensionada, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 51, 9
        )
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blocos = []
        
        for cnt in contours:
            x, y, w_b, h_b = cv2.boundingRect(cnt)
            proporcao_bloco = w_b / float(h_b)
            
            # Filtro geral para pegar caixas de questões e bolhas
            if 15 <= w_b <= 150 and 15 <= h_b <= 150:
                if 0.6 <= proporcao_bloco <= 1.5:
                    # Calcula o preenchimento (caneta)
                    mask = np.zeros(thresh.shape, dtype="uint8")
                    cv2.drawContours(mask, [cnt], -1, 255, -1)
                    densidade = cv2.mean(thresh, mask=mask)[0]
                    
                    blocos.append({
                        'x': x, 'y': y, 'w': w_b, 'h': h_b,
                        'cx': x + w_b/2.0, 'cy': y + h_b/2.0,
                        'densidade': densidade
                    })
        
        if len(blocos) == 0:
            st.warning("Nenhum bloco de marcação estruturado foi localizado na imagem.")
        else:
            # 4. AGRUPAMENTO POR LINHAS HORIZONTAIS
            blocos.sort(key=lambda b: b['cy'])
            linhas = []
            linha_atual = [blocos[0]]
            limite_distancia_y = 25  # Pixels de tolerância vertical
            
            for b in blocos[1:]:
                media_y = np.mean([item['cy'] for item in linha_atual])
                if abs(b['cy'] - media_y) <= limite_distancia_y:
                    linha_atual.append(b)
                else:
                    linhas.append(linha_atual)
                    linha_atual = [b]
            linhas.append(linha_atual)
            
            # Filtra apenas linhas válidas (Número da questão + alternativas)
            linhas = [r for r in linhas if len(r) >= 2]
            linhas.sort(key=lambda r: np.mean([item['cy'] for item in r]))
            
            # 5. MAPEAMENTO DE QUESTÕES E RESPOSTAS
            letras = ['A', 'B', 'C', 'D', 'E', 'F']
            dados_finais = []
            img_feedback = img_redimensionada.copy()
            
            for idx, linha in enumerate(linhas):
                # Ordena da esquerda para a direita
                linha.sort(key=lambda b: b['cx'])
                
                # O elemento mais à esquerda é SIMPRE o número da questão!
                numero_questao_box = linha[0]
                alternativas_box = linha[1:]
                
                # Desenha o Retângulo AZUL no número da questão
                cv2.rectangle(
                    img_feedback, 
                    (int(numero_questao_box['x']), int(numero_questao_box['y'])),
                    (int(numero_questao_box['x'] + numero_questao_box['w']), int(numero_questao_box['y'] + numero_questao_box['h'])), 
                    (255, 0, 0), 2
                )
                
                # Procura qual alternativa está mais preenchida de caneta
                maior_densidade = -1
                letra_escolhida = "NULO"
                bloco_escolhido = None
                
                for opt_idx, alt in enumerate(alternatives_box[:5]): # Limita a 5 opções (A até E)
                    if alt['densidade'] > maior_densidade and alt['densidade'] > 65:
                        maior_densidade = alt['densidade']
                        letra_escolhida = letras[opt_idx]
                        bloco_escolhido = alt
                
                # Se achou uma resposta marcada, desenha em VERDE
                if bloco_escolhido:
                    cv2.rectangle(
                        img_feedback, 
                        (int(bloco_escolhido['x']), int(bloco_escolhido['y'])),
                        (int(bloco_escolhido['x'] + bloco_escolhido['w']), int(bloco_escolhido['y'] + bloco_escolhido['h'])), 
                        (0, 200, 0), 2
                    )
                
                dados_finais.append({
                    "Questão": f"Questão {idx + 1}",
                    "Resposta Detectada": letra_escolhida
                })
            
            # ==========================================
            # EXIBIÇÃO DOS RESULTADOS NA TELA
            # ==========================================
            col1, col2 = st.columns([1.3, 1])
            
            with col1:
                st.subheader("🔍 Visualização do Mapeamento")
                st.caption("Legenda: Quadrados Azuis = Números das Questões | Quadrados Verdes = Resposta Marcada")
                st.image(cv2.cvtColor(img_feedback, cv2.COLOR_BGR2RGB), use_container_width=True)
                
            with col2:
                st.subheader("📋 Gabarito Extraído")
                df_resultados = pd.DataFrame(dados_finais)
                st.dataframe(df_resultados, use_container_width=True, hide_index=True)
