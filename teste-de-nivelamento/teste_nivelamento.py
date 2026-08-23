
import random
import math
import unittest
import pandas as pd

def gerar_datasets():
    """Gera duas listas de dicionários simulando tabelas de um banco de dados."""
    # A seed fixa ajuda na reprodutibilidade, mas os testes abaixo não dependem
    # de valores fixados no código (hardcoded), usando o Pandas para validação.
    random.seed(42)
    n = 200
    ids = list(range(1, n + 1))

    tabela_notas = []
    for i in ids:
        tabela_notas.append({
            'aluno_id': i,
            'horas_estudo': round(random.gauss(10, 3), 1),
            'nota_programacao': round(random.gauss(7, 1.5), 1),
            'turno': random.choice(['Matutino', 'Noturno'])
        })

    # Inserindo valores nulos (None) propositais
    for i in range(10, 20):
        tabela_notas[i]['nota_programacao'] = None

    ids_embaralhados = ids[:]
    random.shuffle(ids_embaralhados)
    tabela_presenca = []
    for i in ids_embaralhados:
        tabela_presenca.append({
            'aluno_id': i,
            'presenca': round(random.gauss(80, 10), 1)
        })

    return tabela_notas, tabela_presenca

# =======================================================
# ÁREA DO ALUNO - PREENCHA AS FUNÇÕES ABAIXO EM PYTHON PURO
# =======================================================

def questao_1_manipular_dados(tb_notas, tb_presenca):
    """
    Realize um INNER JOIN manual entre as duas listas de dicionários
    usando a chave 'aluno_id'.
    Filtre o resultado mantendo APENAS alunos com 'presenca' >= 75.0.
    Retorne uma nova lista de dicionários com os dados combinados.
    """
    mapa_presenca = {p['aluno_id']: p['presenca'] for p in tb_presenca}

    resultado = []
    for nota in tb_notas:
      aluno_id = nota['aluno_id']
      if aluno_id in mapa_presenca:
        presenca = mapa_presenca[aluno_id]
        if presenca >= 75.0:
          novo_registro = nota.copy()
          novo_registro['presenca'] = presenca
          resultado.append(novo_registro)

    return resultado

def questao_2_estatistica(tb_notas):
    """
    Calcule a média e o desvio padrão amostral da chave 'horas_estudo'.
    Dica: o desvio padrão amostral divide pela quantidade (N - 1).
    Retorne a tupla: (media, desvio_padrao).
    """
    valores = [item['horas_estudo'] for item in tb_notas if item['horas_estudo'] is not None]
    n = len(valores)

    if n == 0:
      return 0.0, 0.0

    media = sum(valores) / n

    if n > 1:
      soma_diff_quadrada = sum((x - media) ** 2 for x in valores)
      desvio_padrao = math.sqrt(soma_diff_quadrada / (n-1))
    else:
      desvio_padrao = 0.0

    return media, desvio_padrao

def questao_3_divisao_manual(dados, proporcao_teste=0.2):
    """
    Divida a lista de dados em treino e teste MANUALMENTE usando fatiamento (slicing).
    Separe os primeiros elementos para treino e o final para teste.
    Retorne a tupla: (lista_treino, lista_teste).
    """
    n = len(dados)
    tamanho_teste = int(n * proporcao_teste)
    ponto_corte = n - tamanho_teste

    lista_treino = dados[:ponto_corte]
    lista_teste = dados[ponto_corte:]

    return lista_treino, lista_teste

def questao_4_dados_nulos(tb_notas):
    """
    Identifique os valores nulos (None) na chave 'nota_programacao'
    e substitua-os pela média das notas válidas desta mesma chave.
    Retorne a lista atualizada.
    """
    notas_validas = [d['nota_programacao'] for d in tb_notas if d['nota_programacao'] is not None]
    media = sum(notas_validas) / len(notas_validas) if notas_validas else 0.0

    for d in tb_notas:
      if d['nota_programacao'] is None:
        d['nota_programacao'] = media

    return tb_notas

def questao_5_calculo_erro(y_true, y_pred):
    """
    Recebe duas listas de mesmo tamanho: valores reais e predições.
    Calcule o Erro Quadrático Médio (MSE).
    Retorne o valor numérico do MSE.
    """
    n = len(y_true)
    if n == 0:
        return 0.0
    
    soma_erros_quadrados = sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred))
    mse = soma_erros_quadrados / n
    return mse

def questao_6_agrupamento(tb_notas):
    """
    Agrupe os dados pela chave 'turno' e calcule a média de
    'horas_estudo' para cada turno.
    Retorne um dicionário no formato: {'Matutino': media, 'Noturno': media}.
    """
    grupos = {}
    
    for d in tb_notas:
        turno = d['turno']
        horas = d['horas_estudo']
        if turno not in grupos:
            grupos[turno] = []
        grupos[turno].append(horas)
        
    resultado = {turno: (sum(horas) / len(horas)) for turno, horas in grupos.items()}
    return resultado

# =======================================================
# TESTES DE VALIDAÇÃO AUTOMÁTICA (NÃO ALTERE ESTA PARTE)
# =======================================================
class TestNivelamentoPythonPuro(unittest.TestCase):
    def setUp(self):
        # Gerando cópias novas para cada teste
        self.tb_notas, self.tb_presenca = gerar_datasets()
        # Convertendo para DataFrames estritamente para o cálculo do gabarito dinâmico
        self.df_notas = pd.DataFrame(self.tb_notas)
        self.df_presenca = pd.DataFrame(self.tb_presenca)

    def test_q1_manipulacao(self):
        try:
            # Gabarito Dinâmico
            df_merged = pd.merge(self.df_notas, self.df_presenca, on='aluno_id', how='inner')
            df_esperado = df_merged[df_merged['presenca'] >= 75.0]

            resultado = questao_1_manipular_dados(self.tb_notas, self.tb_presenca)

            self.assertIsNotNone(resultado)
            self.assertTrue(isinstance(resultado, list))
            self.assertEqual(len(resultado), len(df_esperado), "A quantidade de alunos após o join e filtro está incorreta.")
            self.assertIn('presenca', resultado[0], "A chave 'presenca' não foi encontrada após o Join.")
            self.assertTrue(all(d.get('presenca', 0) >= 75.0 for d in resultado), "Há alunos com presença inferior a 75.0.")
        except Exception as e: self.fail(f"Erro na Questão 1: {e}")

    def test_q2_estatistica(self):
        try:
            # Gabarito Dinâmico
            esperado_media = self.df_notas['horas_estudo'].mean()
            esperado_std = self.df_notas['horas_estudo'].std(ddof=1)

            media, std = questao_2_estatistica(self.tb_notas)

            self.assertAlmostEqual(media, esperado_media, places=2, msg="A média está incorreta.")
            self.assertAlmostEqual(std, esperado_std, places=2, msg="O desvio padrão está incorreto.")
        except Exception as e: self.fail(f"Erro na Questão 2: {e}")

    def test_q3_divisao(self):
        try:
            treino, teste = questao_3_divisao_manual(self.tb_notas, 0.2)

            tamanho_esperado_teste = int(len(self.tb_notas) * 0.2)
            self.assertEqual(len(treino) + len(teste), len(self.tb_notas), "A soma do treino e teste não bate com o total.")
            self.assertEqual(len(teste), tamanho_esperado_teste, "A proporção do conjunto de teste está incorreta.")
        except Exception as e: self.fail(f"Erro na Questão 3: {e}")

    def test_q4_nulos(self):
        try:
            # Gabarito Dinâmico (o Pandas ignora NaNs no cálculo da média automaticamente)
            media_esperada = self.df_notas['nota_programacao'].mean()

            resultado = questao_4_dados_nulos(self.tb_notas)
            nulos = [d for d in resultado if d['nota_programacao'] is None]

            self.assertEqual(len(nulos), 0, "Ainda existem valores nulos (None) na lista.")
            # O índice 15 era sabidamente nulo na geração dos dados
            self.assertAlmostEqual(resultado[15]['nota_programacao'], media_esperada, places=2,
                                   msg="A substituição dos valores nulos não utilizou a média correta.")
        except Exception as e: self.fail(f"Erro na Questão 4: {e}")

    def test_q5_calculo(self):
        try:
            # Gerando listas randômicas para testar
            y_true = [random.uniform(0, 10) for _ in range(15)]
            y_pred = [random.uniform(0, 10) for _ in range(15)]

            # Gabarito Dinâmico
            serie_true = pd.Series(y_true)
            serie_pred = pd.Series(y_pred)
            mse_esperado = ((serie_true - serie_pred) ** 2).mean()

            res = questao_5_calculo_erro(y_true, y_pred)
            self.assertAlmostEqual(res, mse_esperado, places=3, msg="O cálculo do MSE falhou.")
        except Exception as e: self.fail(f"Erro na Questão 5: {e}")

    def test_q6_agrupamento(self):
        try:
            # Gabarito Dinâmico
            agrupado_esperado = self.df_notas.groupby('turno')['horas_estudo'].mean().to_dict()

            res = questao_6_agrupamento(self.tb_notas)

            self.assertTrue(isinstance(res, dict), "O retorno deve ser um dicionário.")
            self.assertIn('Matutino', res)
            self.assertIn('Noturno', res)
            self.assertAlmostEqual(res['Matutino'], agrupado_esperado['Matutino'], places=2)
            self.assertAlmostEqual(res['Noturno'], agrupado_esperado['Noturno'], places=2)
        except Exception as e: self.fail(f"Erro na Questão 6: {e}")

if __name__ == '__main__':
    print("\n=== TESTE DE NIVELAMENTO - AVALIAÇÃO DINÂMICA ===")
    unittest.main(verbosity=2)
