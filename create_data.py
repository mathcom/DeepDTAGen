import os
import argparse
from models.datautils import DeepDTAGenDataProcessor

if __name__ == '__main__':
    # 1. 파서 생성
    parser = argparse.ArgumentParser(description='Data processing for DeepDTAGen')

    # 2. dataset 인자 추가
    # 사용법 예: python main.py --dataset ChEMBL35_IC50
    parser.add_argument(
        '--dataset', 
        type=str, 
        default="ChEMBL35_IC50",
        help='Name of the dataset directory (default: ChEMBL35_IC50)'
    )

    # 3. 인자 파싱
    args = parser.parse_args()
    
    print(f"[*] Starting data processing for dataset: {args.dataset}")

    # 4. 각 폴드별 처리 실행
    for foldname in ['fold1', 'fold2', 'fold3', 'fold4', 'fold5']:
        data_dir = os.path.join('data', args.dataset, foldname)
        
        # 데이터 디렉토리가 실제 존재하는지 체크
        if os.path.exists(data_dir):
            print(f"    Processing {foldname}...")
            processor = DeepDTAGenDataProcessor(data_dir)
            processor.process() # process_datasets가 dataset 이름을 리스트로 받도록 설계된 경우
        else:
            print(f"    [Warning] Directory not found: {data_dir}")