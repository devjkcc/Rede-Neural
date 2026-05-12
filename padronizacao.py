import fitz  # PyMuPDF
from pathlib import Path


path_atual = Path.cwd()
print(path_atual)
# PDF de entrada
PDF_PATH = path_atual / 'pdf' / 'Untitled_20260417_100314 (1) Copy.pdf'

# Pasta de saída
PASTA_SAIDA = path_atual / 'images' / 'imagens_padronizadas'
PASTA_SAIDA.mkdir(exist_ok=True)

# Abrir PDF
pdf = fitz.open(PDF_PATH)

# Percorrer páginas
for numero_pagina in range(len(pdf)):

    pagina = pdf[numero_pagina]

    # Renderizar página
    matriz = fitz.Matrix(1.3, 1.3)
    pix = pagina.get_pixmap(matrix=matriz)

    # Nome do arquivo
    nome_saida = PASTA_SAIDA / f"pagina_{numero_pagina + 1}.png"

    # Salvar imagem
    pix.save(nome_saida)

    print(f"✓ Página {numero_pagina + 1} salva")

print("\n✓ Conversão concluída!")