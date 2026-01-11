import os
from typing import List, Dict, Tuple, Set, Any

import torch
import tqdm
import pandas as pd
import numpy as np
import networkx as nx
from rdkit import Chem

# utils에서 필요한 클래스들이 있다고 가정합니다.
from models.utils import Tokenizer, TestbedDataset 


class MoleculeFeaturizer:
    """분자(SMILES)를 그래프 데이터로 변환하는 클래스"""
    ATOM_SYMBOLS = [
        'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 
        'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 
        'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 
        'Hg', 'Pb', 'Unknown'
    ]
    
    def __init__(self):
        self.atom_set = self.ATOM_SYMBOLS

    def _one_of_k_encoding(self, x: Any, allowable_set: List[Any]) -> List[bool]:
        if x not in allowable_set:
            x = allowable_set[-1]
        return [x == s for s in allowable_set]

    def _one_of_k_encoding_unk(self, x: Any, allowable_set: List[Any]) -> List[bool]:
        if x not in allowable_set:
            x = allowable_set[-1]
        return [x == s for s in allowable_set] + [x not in allowable_set]

    def get_atom_features(self, atom: Chem.Atom) -> np.ndarray:
        return np.array(
            self._one_of_k_encoding_unk(atom.GetSymbol(), self.atom_set) +
            self._one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
            self._one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
            self._one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
            self._one_of_k_encoding_unk(atom.GetFormalCharge(), [-1, -2, 1, 2, 0]) +
            self._one_of_k_encoding_unk(atom.GetHybridization(), [
                Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
                Chem.rdchem.HybridizationType.SP3D2
            ]) +
            [atom.GetIsAromatic()] +
            [atom.IsInRing()]
        )


    def get_bond_features(self, bond: Chem.Bond) -> np.ndarray:
        bt = bond.GetBondType()
        # [Single, Double, Triple, Aromatic, BondTypeAsDouble]
        bond_feats = [0, 0, 0, 0, bond.GetBondTypeAsDouble()]
        if bt == Chem.rdchem.BondType.SINGLE:
            bond_feats[0] = 1
        elif bt == Chem.rdchem.BondType.DOUBLE:
            bond_feats[1] = 1
        elif bt == Chem.rdchem.BondType.TRIPLE:
            bond_feats[2] = 1
        elif bt == Chem.rdchem.BondType.AROMATIC:
            bond_feats[3] = 1
        return np.array(bond_feats)


    def smile_to_graph(self, smile: str) -> Tuple[int, List[np.ndarray], List[List[int]], List[np.ndarray]]:
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            # RDKit이 파싱 실패 시 예외 처리 혹은 빈 그래프 반환 로직 필요
            return 0, [], [], []

        c_size = mol.GetNumAtoms()

        # Node Features
        features = []
        for atom in mol.GetAtoms():
            feature = self.get_atom_features(atom)
            # Normalize features
            features.append(feature / sum(feature))

        # Edge Features
        edges = []
        for bond in mol.GetBonds():
            edge_feats = self.get_bond_features(bond)
            edges.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), {'edge_feats': edge_feats}))

        g = nx.Graph()
        g.add_edges_from(edges)
        g = g.to_directed()
        
        edge_index = []
        edge_feats = []
        for e1, e2, feats in g.edges(data=True):
            edge_index.append([e1, e2])
            edge_feats.append(feats['edge_feats'])

        return c_size, np.array(features), edge_index, np.array(edge_feats)


class ProteinEncoder:
    """단백질 서열을 정수 시퀀스로 인코딩하는 클래스"""
    SEQ_VOCAB = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
    MAX_SEQ_LEN = 1000


    def __init__(self):
        self.seq_dict = {v: (i + 1) for i, v in enumerate(self.SEQ_VOCAB)}


    def encode(self, prot_seq: str) -> np.ndarray:
        x = np.zeros(self.MAX_SEQ_LEN)
        for i, ch in enumerate(prot_seq[:self.MAX_SEQ_LEN]):
            if ch in self.seq_dict:
                x[i] = self.seq_dict[ch]
        return x


class DeepDTAGenDataProcessor:
    """전체 데이터셋 처리 파이프라인 관리자"""
    def __init__(self, data_dir: str = 'data'):
        self.data_dir = data_dir
        self.processed_dir = os.path.join(data_dir, 'processed')
        os.makedirs(self.processed_dir, exist_ok=True)
        
        self.featurizer = MoleculeFeaturizer()
        self.prot_encoder = ProteinEncoder()
        
        
    def _load_and_aggregate_smiles(self) -> Set[str]:
        """SMILES를 수집하여 Graph 캐싱 준비"""
        all_compound_smiles = []
        for split in ['train', 'valid', 'test']:
            file_path = os.path.join(self.data_dir, f'{split}.csv')
            if os.path.exists(file_path):
                df = pd.read_csv(file_path, low_memory=False)
                all_compound_smiles += list(df['SMILES'])
        return set(all_compound_smiles)
        
        
    def _build_smile_graph_map(self, smiles_set: Set[str]) -> Dict[str, Any]:
        """Unique SMILES에 대해 그래프 미리 계산"""
        print(f"Generating graphs for {len(smiles_set)} unique SMILES...")
        smile_graph = {}
        for smile in tqdm.tqdm(smiles_set):
            g = self.featurizer.smile_to_graph(smile)
            # edge_index 
            if len(g[3]) > 0: smile_graph[smile] = g
        return smile_graph


    def process(self):
        # 1. 전처리 대상 SMILES 수집 및 그래프 변환
        all_smiles = self._load_and_aggregate_smiles()
        smile_graph_map = self._build_smile_graph_map(all_smiles)

        # 2. 데이터셋 전처리 수행
        self._process(smile_graph_map)


    def _process(self, smile_graph_map: Dict[str, Any]):
        train_file = os.path.join(self.processed_dir, 'train.pt')
        valid_file = os.path.join(self.processed_dir, 'valid.pt')
        test_file = os.path.join(self.processed_dir, 'test.pt')
        vocab_file = os.path.join(self.data_dir, 'vocab.txt')

        # 이미 처리된 파일이 있으면 스킵
        if os.path.isfile(train_file) and os.path.isfile(test_file):
            print("Processed files already exist.")
            return

        print(f"Processing...")
        df_train = pd.read_csv(os.path.join(self.data_dir, 'train.csv'), low_memory=False)
        df_valid = pd.read_csv(os.path.join(self.data_dir, 'valid.csv'), low_memory=False)
        df_test = pd.read_csv(os.path.join(self.data_dir, 'test.csv'), low_memory=False)

        # Tokenizer 생성 및 저장 (Train/Test 합집합 기준)
        all_target_smiles = set(df_train['SMILES']).union(set(df_test['SMILES']))
        vocab = Tokenizer.gen_vocabs(list(all_target_smiles))
        tokenizer = Tokenizer(vocab)
        
        with open(vocab_file, 'w') as file:
            for v in sorted(list(vocab)):
                file.write(f"{v}\n")
        
        # Train 데이터 처리
        train_data = self._prepare_data_object(df_train, tokenizer, smile_graph_map, 'train')
        torch.save((train_data.data, train_data.slices), train_file)
        
        # Valid 데이터 처리
        valid_data = self._prepare_data_object(df_valid, tokenizer, smile_graph_map, 'valid')
        torch.save((valid_data.data, valid_data.slices), valid_file)
        
        # Test 데이터 처리
        test_data = self._prepare_data_object(df_test, tokenizer, smile_graph_map, 'test')
        torch.save((test_data.data, test_data.slices), test_file)

        print(f"Created {train_file}, {valid_file}, and {test_file}")


    def _prepare_data_object(self, df: pd.DataFrame, tokenizer: Tokenizer, 
                             smile_graph_map: Dict, dataset_tag: str):
        # 데이터 추출
        # SMILES: Graph 입력
        # target_smiles (MTS): Generative Model의 타겟 시퀀스
        compound_smiles = list(df['SMILES'])
        target_smiles = list(df['SMILES']) 
        target_sequences = list(df['FASTA'])
        affinities = list(df['affinity'])

        # 단백질 인코딩
        encoded_prots = [self.prot_encoder.encode(t) for t in target_sequences]

        # 타겟 스마일즈 토크나이징 (Tokenizer가 리스트 반환한다고 가정)
        tokenized_smiles = [torch.LongTensor(tokenizer.parse(s)) for s in target_smiles]

        # Numpy 변환
        compound_smiles = np.asarray(compound_smiles)
        target_smiles = np.asarray(target_smiles)
        encoded_prots = np.asarray(encoded_prots)
        affinities = np.asarray(affinities)

        # PyTorch Geometric Dataset 생성 (utils.TestbedDataset 사용)
        return TestbedDataset(
            root=self.data_dir,
            dataset=dataset_tag,
            xd=compound_smiles,
            xdt=tokenized_smiles,
            xt=encoded_prots,
            y=affinities,
            smile_graph=smile_graph_map
        )


if __name__ == "__main__":
    # 실행부
    processor = DeepDTAGenDataProcessor(data_dir='data')
    processor.process()