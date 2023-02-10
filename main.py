import urllib.request
import io
import zipfile
import networkx as nx
import pandas as pd
import tensorflow_gnn as tfgnn
import numpy as np

print("start")

url = "http://www-personal.umich.edu/~mejn/netdata/football.zip"
sock = urllib.request.urlopen(url)  # open URL
s = io.BytesIO(sock.read())  # read into BytesIO "file"
sock.close()

zf = zipfile.ZipFile(s)  # zipfile object
txt = zf.read("football.txt").decode()  # read info file
gml = zf.read("football.gml").decode()  # read gml data
# throw away bogus first line with # from mejn files
gml = gml.split("\n")[1:]
G = nx.parse_gml(gml)  # parse gml data


# print(txt)
# print(gml)


cmap = {0:'#bd2309', 1:'#bbb12d',2:'#1480fa',3:'#14fa2f',4:'#faf214',
        5:'#2edfea',6:'#ea2ec4',7:'#ea2e40',8:'#577a4d',9:'#2e46c0',
        10:'#f59422',11:'#8086d9'}

colors = [cmap[G.nodes[n]['value']] for n in G.nodes()]
pos = nx.spring_layout(G, seed=1987)
nx.draw_networkx_edges(G, pos, alpha=0.2)
nx.draw_networkx_nodes(G, pos, nodelist=G.nodes(), node_color=colors, node_size=100)


node_data = G.nodes(data=True)
edge_data = G.edges(data=True)


# Convert to pandas
node_df = pd.DataFrame.from_dict(dict(G.nodes(data=True)), orient='index')
node_df.index.name = 'school'
node_df.columns = ['conference']
edge_df = nx.to_pandas_edgelist(G)


# test cases splitting
from sklearn.model_selection import train_test_split
node_train, node_test = train_test_split(node_df,test_size=0.15,random_state=42)
edge_train = edge_df.loc[~((edge_df['source'].isin(node_test.index)) | (edge_df['target'].isin(node_test.index)))]
edge_test = edge_df.loc[(edge_df['source'].isin(node_test.index)) | (edge_df['target'].isin(node_test.index))]





def bidirectional(edge_df):
  reverse_df = edge_df.rename(columns={'source':'target','target':'source'})
  reverse_df = reverse_df[edge_df.columns]
  reverse_df = pd.concat([edge_df, reverse_df], ignore_index=True, axis=0)
  return reverse_df

def create_adj_id(node_df,edge_df):
  node_df = node_df.reset_index().reset_index()
  edge_df = pd.merge(edge_df,node_df[['school','index']].rename(columns={"index":"source_id"}),
                     how='left',left_on='source',right_on='school').drop(columns=['school'])
  edge_df = pd.merge(edge_df,node_df[['school','index']].rename(columns={"index":"target_id"}),
                     how='left',left_on='target',right_on='school').drop(columns=['school'])

  edge_df.dropna(inplace=True)
  return node_df, edge_df

edge_full_adj = bidirectional(edge_df)
edge_train_adj = bidirectional(edge_train)

node_full_adj,edge_full_adj = create_adj_id(node_df,edge_full_adj)
node_train_adj,edge_train_adj = create_adj_id(node_train,edge_train_adj)


def create_graph_tensor(node_df,edge_df):
  graph_tensor = tfgnn.GraphTensor.from_pieces(
      node_sets = {
          "schools": tfgnn.NodeSet.from_fields(
              sizes = [len(node_df)],
              features ={
                  'Latitude': np.array(node_df['Latitude'], dtype='float32').reshape(len(node_df),1),
                  'Longitude': np.array(node_df['Longitude'], dtype='float32').reshape(len(node_df),1),
                  'Rank': np.array(node_df['Rank'], dtype='int32').reshape(len(node_df),1),
                  'Wins': np.array(node_df['Wins'], dtype='int32').reshape(len(node_df),1),
                  'Conf_wins': np.array(node_df['Conf_wins'], dtype='int32').reshape(len(node_df),1),
                  'conference': np.array(node_df.iloc[:,-12:], dtype='int32'),
              }),
      },
      edge_sets ={
          "games": tfgnn.EdgeSet.from_fields(
              sizes = [len(edge_df)],
              features = {
                  'name_sim_score': np.array(edge_df['name_sim_score'], dtype='float32').reshape(len(edge_df),1),
                  'euclidean_dist': np.array(edge_df['euclidean_dist'], dtype='float32').reshape(len(edge_df),1),
                  'conference_game': np.array(edge_df['conference_game'], dtype='int32').reshape(len(edge_df),1)
              },
              adjacency = tfgnn.Adjacency.from_indices(
                  source = ("schools", np.array(edge_df['source_id'], dtype='int32')),
                  target = ("schools", np.array(edge_df['target_id'], dtype='int32')),
              )),
      })
  return graph_tensor

print(full_tensor)

full_tensor = create_graph_tensor(node_full_adj,edge_full_adj)
train_tensor = create_graph_tensor(node_train_adj,edge_train_adj)



print('this is the end of it')