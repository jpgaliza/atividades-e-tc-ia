"""
Teste: implementação de uma rede neural já treinada (propagação direta)
Matemática Computacional - Prof. Dr. Alexsandro O. A.

Nome(s): ____________________________________________

Regras:
- Use apenas Python, NumPy e pandas.
- É proibido usar PyTorch, TensorFlow, Keras, sklearn ou qualquer rede neural pronta.
- Implemente as funções marcadas com "TODO" (remova o `raise NotImplementedError`).
- Rode `python rede_template.py`: o teste de validação mostra OK ou FALHOU.
"""

import json
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Leitura dos pesos (pronta)
# ---------------------------------------------------------------------------

def carregar_rede(arquivo="pesos_rede.json"):
    """Lê o JSON e devolve um dicionário com os pesos como arrays NumPy."""
    with open(arquivo, encoding="utf-8") as f:
        cfg = json.load(f)
    for chave in ["W1", "b1", "W2", "b2", "W3", "b3", "normalizacao_divisor"]:
        cfg[chave] = np.array(cfg[chave], dtype=float)
    return cfg


# ---------------------------------------------------------------------------
# Funções a implementar
# ---------------------------------------------------------------------------

def normalizar(entrega_bruta, divisores):
    """TODO: divida cada atributo bruto pelo divisor correspondente.

    entrega_bruta = [distancia_km, peso_kg, urgencia, valor_declarado]
    Retorne o vetor x (tamanho 4) que entra na rede.
    """
    return np.array(entrega_bruta, dtype=float) / np.asarray(divisores, dtype=float)

def relu(z):
    """TODO: ReLU(z) = max(0, z), elemento a elemento."""
    return np.maximum(0, z)


def tanh(z):
    """TODO: tangente hiperbólica, elemento a elemento (pode usar np.tanh)."""
    return np.tanh(z)


def softmax(z):
    """TODO: softmax(z)_i = exp(z_i - max(z)) / soma_j exp(z_j - max(z)).

    Subtrair max(z) não muda o resultado e evita overflow.
    """
    z = np.asarray(z, dtype=float)
    exp_z = np.exp(z - np.max(z))
    return exp_z / np.sum(exp_z)


def camada(W, b, x, ativacao):
    """TODO: devolva (z, h), com z = W @ x + b e h = ativacao(z)."""
    z = W @ x + b
    h = ativacao(z)
    return z, h


def propagar(rede, x):
    """TODO: faça a propagação direta pelas 3 camadas de pesos.

    Retorne um dicionário com as chaves
    "z1", "h1", "z2", "h2", "z3", "p"  (p = probabilidades da saída).
    Ativações: camada 1 -> ReLU, camada 2 -> tanh, camada 3 -> softmax.
    """
    z1, h1 = camada(rede["W1"], rede["b1"], x, relu)
    z2, h2 = camada(rede["W2"], rede["b2"], h1, tanh)
    z3, p = camada(rede["W3"], rede["b3"], h2, softmax)
    return {
        "z1": z1,
        "h1": h1,
        "z2": z2,
        "h2": h2,
        "z3": z3,
        "p": p,
    }


def classificar(rede, x):
    """TODO: devolva (nome_da_classe, probabilidade) da classe mais provável."""
    p = propagar(rede, x)["p"]
    i = int(np.argmax(p))
    return rede["classes"][i], float(p[i])


# ---------------------------------------------------------------------------
# Teste de validação (exemplo resolvido no enunciado)
# ---------------------------------------------------------------------------

def _confere(nome, obtido, esperado, tol=1e-3):
    ok = np.allclose(np.asarray(obtido, float), np.asarray(esperado, float), atol=tol)
    print(f"[{'OK' if ok else 'FALHOU'}] {nome}")
    if not ok:
        print("    esperado:", np.round(esperado, 4), "\n    obtido:  ", np.round(obtido, 4))


def validar(rede):
    print("=" * 60 + "\nVALIDAÇÃO: x = (0.60, 0.40, 0.50, 0.20)\n" + "=" * 60)
    x = np.array([0.60, 0.40, 0.50, 0.20])
    r = propagar(rede, x)
    _confere("z1", r["z1"], [1.602, -0.417, 1.433, -1.171, -0.521, 1.274])
    _confere("h1", r["h1"], [1.602, 0.000, 1.433, 0.000, 0.000, 1.274])
    _confere("z2", r["z2"], [2.5760, -0.9262, 4.1550, 0.0951])
    _confere("h2", r["h2"], [0.9885, -0.7288, 0.9995, 0.0948])
    _confere("z3", r["z3"], [-3.5755, 4.4592, -1.0141])
    _confere("p", r["p"], [0.0003, 0.9955, 0.0042])
    classe, prob = classificar(rede, x)
    print(f"Classe prevista: {classe} ({100 * prob:.2f}%)  -> esperado: SEDEX (99.55%)")


# ---------------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        rede = carregar_rede()
        validar(rede)

        # Use as linhas abaixo para responder ao quiz.
        np.set_printoptions(precision=4, suppress=True)
        print("\nNúmero de parâmetros:",
              sum(rede[k].size for k in ["W1", "b1", "W2", "b2", "W3", "b3"]))

        df = pd.read_csv("entregas_teste_rede.csv")
        print("\nEntregas de teste:")
        for _, linha in df.iterrows():
            bruta = [linha.distancia_km, linha.peso_kg, linha.urgencia, linha.valor_declarado]
            x = normalizar(bruta, rede["normalizacao_divisor"])
            classe, prob = classificar(rede, x)
            print(f"  entrega {linha.id_entrega:2d}: prevista = {classe:8s} ({100 * prob:6.2f}%)"
                  f" | real = {linha.modalidade_real}")
    except NotImplementedError as e:
        print(f"\nFunção ainda não implementada: {e}()")
