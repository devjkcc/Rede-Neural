import pandas as pd
import cv2
import numpy as np

from utils.preprocess import preprocess_image
from utils.regions import REGIOES
from utils.trocr import reconhecer_texto
from utils.checkbox import checkbox_marcado
from utils.easyocr import reconhecer_numeros

from pathlib import Path

# =========================
# CAMINHOS
# =========================

path_atual = Path.cwd()
path_anterior_1 = path_atual.parent

caminho_imagens = (
    path_atual / 'images' / 'imagens_padronizadas'
)

# =========================
# CONFIGURAÇÕES
# =========================

MARGEM_EXTRA = 2  # Margem extra para buscar campos (em pixels)

# =========================
# IMAGEM
# =========================

# Pegar todas as imagens da pasta
imagens = sorted([f for f in caminho_imagens.glob("*.png")])

resultados = []

# =========================
# FUNÇÃO PARA ENCONTRAR TRANSFORMAÇÃO (CORRIGIDA)
# =========================

def encontrar_transformacao_imagem(img_atual, img_referencia=None):
    """
    Encontra a transformação (escala, rotação, translação) entre imagens.
    Usa Homografia + RANSAC para robustez e valida offsets absurdos.
    """
    try:
        if img_referencia is None:
            return np.eye(3, 3, dtype=np.float32), (0, 0)
        
        # Converter para escala de cinza
        gray_atual = cv2.cvtColor(img_atual, cv2.COLOR_BGR2GRAY) if len(img_atual.shape) == 3 else img_atual
        gray_ref = cv2.cvtColor(img_referencia, cv2.COLOR_BGR2GRAY) if len(img_referencia.shape) == 3 else img_referencia
        
        # Detector ORB
        orb = cv2.ORB_create(nfeatures=500)
        kp1, des1 = orb.detectAndCompute(gray_ref, None)
        kp2, des2 = orb.detectAndCompute(gray_atual, None)
        
        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            return np.eye(3, 3, dtype=np.float32), (0, 0)
        
        # Matcher com Lowe's ratio test
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(des1, des2, k=2)
        
        # Filtrar bons matches
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) < 4:
            print(f"  ⚠️  Poucos matches ({len(good_matches)}), usando identidade")
            return np.eye(3, 3, dtype=np.float32), (0, 0)
        
        # Extrair pontos
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Usar Homografia com RANSAC
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if H is None:
            return np.eye(3, 3, dtype=np.float32), (0, 0)
        
        # Extrair translação
        offset_x = int(H[0, 2])
        offset_y = int(H[1, 2])
        
        # VALIDAÇÃO: Rejeitar offsets muito grandes
        img_height, img_width = img_atual.shape[:2]
        max_offset = max(img_height, img_width) * 0.15  # 15% do tamanho
        
        if abs(offset_x) > max_offset or abs(offset_y) > max_offset:
            print(f"  ⚠️  Offset absurdo rejeitado: ({offset_x}, {offset_y}), usando identidade")
            return np.eye(3, 3, dtype=np.float32), (0, 0)
        
        print(f"  ✓ Transformação OK - Offset: ({offset_x}, {offset_y})")
        return H, (offset_x, offset_y)
        
    except Exception as e:
        print(f"  ⚠️  Erro na transformação: {e}")
        return np.eye(3, 3, dtype=np.float32), (0, 0)


def aplicar_transformacao_coord(x, y, H):
    """Aplica transformação homográfica a uma coordenada"""
    if H.shape == (3, 3):
        ponto = np.array([[[x, y]]], dtype=np.float32)
        resultado = cv2.perspectiveTransform(ponto, H)
        return int(resultado[0, 0, 0]), int(resultado[0, 0, 1])
    else:
        # Se for matriz 2x3 (afim)
        return int(x + H[0, 2]), int(y + H[1, 2])

# =========================
# FUNÇÃO OCR
# =========================

def processar_campo_texto(nome_regiao, nome_debug, M=None):

    x1, y1, x2, y2 = REGIOES[nome_regiao]
    
    # Converter para int
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    # Aplicar transformação se disponível
    if M is not None:
        x1, y1 = aplicar_transformacao_coord(x1, y1, M)
        x2, y2 = aplicar_transformacao_coord(x2, y2, M)
    
    # Garantir que as coordenadas estão dentro da imagem
    y1 = int(max(0, y1))
    y2 = int(min(img.shape[0], y2))
    x1 = int(max(0, x1))
    x2 = int(min(img.shape[1], x2))

    # Validação: crop precisa ter altura e largura > 0
    if y2 <= y1 or x2 <= x1:
        print(f"    ⚠️  Coordenadas inválidas para {nome_debug}: ({x1},{y1}) -> ({x2},{y2})")
        return ""

    crop = img[y1:y2, x1:x2]

    # padding melhora MUITO transformer OCR
    crop = cv2.copyMakeBorder(
        crop,
        20,
        20,
        20,
        20,
        cv2.BORDER_CONSTANT,
        value=[255,255,255]
    )

    # aumenta resolução
    crop = cv2.resize(
        crop,
        None,
        fx=5,
        fy=5,
        interpolation=cv2.INTER_CUBIC
    )

    # blur leve
    crop = cv2.GaussianBlur(
        crop,
        (3,3),
        0
    )

    # debug
    cv2.imwrite(
        f"debug_{nome_debug}.png",
        crop
    )

    # OCR
    texto = reconhecer_texto(crop)

    return texto

# =========================
# FUNÇÃO CHECKBOX
# =========================

def processar_checkboxes(nome_regiao, nome_debug, M=None):

    selecionados = []

    for item, coords in REGIOES[nome_regiao].items():

        x1, y1, x2, y2 = coords
        
        # Converter para int
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Aplicar transformação se disponível
        if M is not None:
            x1, y1 = aplicar_transformacao_coord(x1, y1, M)
            x2, y2 = aplicar_transformacao_coord(x2, y2, M)

        # Adicionar margem de segurança
        y1 = int(max(0, y1 - MARGEM_EXTRA))
        y2 = int(min(img.shape[0], y2 + MARGEM_EXTRA))

        x1 = int(max(0, x1 - MARGEM_EXTRA))
        x2 = int(min(img.shape[1], x2 + MARGEM_EXTRA))

        # Validação: crop precisa ter altura e largura > 0
        if y2 <= y1 or x2 <= x1:
            print(f"    ⚠️  Checkbox {item} ignorado - coordenadas inválidas")
            continue

        crop = img[y1:y2, x1:x2]

        # debug
        crop_debug = cv2.resize(
            crop,
            None,
            fx=8,
            fy=8,
            interpolation=cv2.INTER_CUBIC
        )

        cv2.imwrite(
            f"debug_{nome_debug}_{item}.png",
            crop_debug
        )

        if checkbox_marcado(crop):
            selecionados.append(item)

    return selecionados

# =========================
# FUNÇÃO TELEFONE
# =========================

def processar_telefone(M=None):

    x1, y1, x2, y2 = REGIOES["telefone"]
    
    # Converter para int
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    # Aplicar transformação se disponível
    if M is not None:
        x1, y1 = aplicar_transformacao_coord(x1, y1, M)
        x2, y2 = aplicar_transformacao_coord(x2, y2, M)
    
    # Garantir que as coordenadas estão dentro da imagem
    y1 = int(max(0, y1))
    y2 = int(min(img.shape[0], y2))
    x1 = int(max(0, x1))
    x2 = int(min(img.shape[1], x2))

    # Validação
    if y2 <= y1 or x2 <= x1:
        print(f"    ⚠️  Coordenadas inválidas para telefone")
        return ""

    crop = img[y1:y2, x1:x2]

    # padding ajuda MUITO no TrOCR
    crop = cv2.copyMakeBorder(
        crop,
        20,
        20,
        20,
        20,
        cv2.BORDER_CONSTANT,
        value=[255,255,255]
    )

    # aumenta resolução
    crop = cv2.resize(
        crop,
        None,
        fx=5,
        fy=5,
        interpolation=cv2.INTER_CUBIC
    )

    # blur leve
    crop = cv2.GaussianBlur(
        crop,
        (3,3),
        0
    )

    # debug
    cv2.imwrite(
        "debug_telefone.png",
        crop
    )

    # TrOCR
    texto = reconhecer_texto(crop)

    return texto


def processar_numerico(nome_regiao, nome_debug, M=None):

    x1, y1, x2, y2 = REGIOES[nome_regiao]
    
    # Converter para int
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    # Aplicar transformação se disponível
    if M is not None:
        x1, y1 = aplicar_transformacao_coord(x1, y1, M)
        x2, y2 = aplicar_transformacao_coord(x2, y2, M)
    
    # Garantir que as coordenadas estão dentro da imagem
    y1 = int(max(0, y1))
    y2 = int(min(img.shape[0], y2))
    x1 = int(max(0, x1))
    x2 = int(min(img.shape[1], x2))

    # Validação
    if y2 <= y1 or x2 <= x1:
        print(f"    ⚠️  Coordenadas inválidas para {nome_debug}")
        return ""

    crop = img[y1:y2, x1:x2]

    # remove linha inferior
    crop = crop[:-2, :]

    # padding
    crop = cv2.copyMakeBorder(
        crop,
        5,
        5,
        5,
        5,
        cv2.BORDER_CONSTANT,
        value=[255,255,255]
    )

    # aumenta MUITO
    crop = cv2.resize(
        crop,
        None,
        fx=3,
        fy=3,
        interpolation=cv2.INTER_CUBIC
    )

    # blur leve
    crop = cv2.GaussianBlur(
        crop,
        (3,3),
        0
    )

    # debug
    cv2.imwrite(
        f"debug_{nome_debug}.png",
        crop
    )

    texto = reconhecer_numeros(crop)

    return texto

# =========================
# PROCESSAMENTO DE TODAS AS IMAGENS
# =========================

# Carregar imagem de referência fixa
img_referencia = cv2.imread("referencia.png")
if img_referencia is None:
    print("⚠️  AVISO: 'referencia.png' não encontrado! Será usada a primeira imagem como referência.")
    img_referencia = None

for idx, caminho_imagem in enumerate(imagens):
    print(f"\n📷 [{idx+1}/{len(imagens)}] Processando: {caminho_imagem.name}")
    
    img, thresh = preprocess_image(caminho_imagem)
    
    # Usar referência fixa ou primeira imagem se não existir
    if img_referencia is not None:
        M, (offset_x, offset_y) = encontrar_transformacao_imagem(img, img_referencia)
    elif idx == 0:
        # Se não houver referência, usar primeira imagem
        M = np.eye(3, 3, dtype=np.float32)
        print(f"  ℹ️  Usando primeira imagem como calibração")
    else:
        M, (offset_x, offset_y) = encontrar_transformacao_imagem(img, img)
    
    resultado = {}
    resultado["nome_arquivo"] = caminho_imagem.name

    # =========================
    # CLIENTE
    # =========================

    resultado["cliente"] = processar_campo_texto(
        "cliente",
        "cliente",
        M
    )

    # =========================
    # CODIGO
    # =========================

    resultado["codigo"] = processar_numerico(
        "codigo",
        "codigo",
        M
    )

    # =========================
    # TELEFONE
    # =========================

    resultado["telefone"] = processar_numerico(
        "telefone",
        "telefone",
        M
    )

    # =========================
    # CHECKBOXES
    # =========================

    resultado["dias_consumo"] = processar_checkboxes(
        "dias_consumo",
        "dias_consumo",
        M
    )

    resultado["dias_recebimento"] = processar_checkboxes(
        "dias_recebimento",
        "dias_recebimento",
        M
    )

    resultado["periodo_recebimento"] = processar_checkboxes(
        "periodo_recebimento",
        "periodo",
        M
    )

    resultado["veiculos_permitidos"] = processar_checkboxes(
        "veiculos_permitidos",
        "veiculo",
        M
    )

    resultado["necessidade_reservar_vaga"] = processar_checkboxes(
        "necessidade_reservar_vaga",
        "vaga",
        M
    )

    resultado["visibilidade_trajeto"] = processar_checkboxes(
        "visibilidade_trajeto",
        "trajeto",
        M
    )

    resultado["local_facil_acesso"] = processar_checkboxes(
        "local_facil_acesso",
        "acesso",
        M
    )

    resultado["possui_estacionamento"] = processar_checkboxes(
        "possui_estacionamento",
        "estacionamento",
        M
    )
    
    # Adicionar resultado à lista
    resultados.append(resultado)
    print(f"✓ {caminho_imagem.name} processado!")

# =========================
# RESULTADO
# =========================

print(f"\nTotal de imagens processadas: {len(resultados)}")
print(resultados)

# =========================
# EXPORTAR
# =========================

df = pd.DataFrame(resultados)

df.to_excel(
    "output/resultado.xlsx",
    index=False
)

print("✓ Excel gerado com sucesso!")