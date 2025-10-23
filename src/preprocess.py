# src/preprocess.py (اصلاح شده)
import pyshark
import pandas as pd

def extract_features(pcap_file):
    cap = pyshark.FileCapture(pcap_file)
    data = []
    for pkt in cap:
        protocol = pkt.highest_layer if hasattr(pkt, 'highest_layer') else 'Unknown'
        length = int(pkt.length) if hasattr(pkt, 'length') else 0
        src = pkt.ip.src if 'IP' in pkt else '0.0.0.0'
        dst = pkt.ip.dst if 'IP' in pkt else '0.0.0.0'
        data.append({
            'protocol': protocol,
            'length': length,
            'src': src,
            'dst': dst
        })
    
    df = pd.DataFrame(data)
    
    # فقط اگر ستون protocol وجود داشت کدگذاری کن
    if 'protocol' in df.columns:
        df['protocol'] = df['protocol'].astype('category').cat.codes
    df['src'] = df['src'].astype('category').cat.codes
    df['dst'] = df['dst'].astype('category').cat.codes
    df['length'] = df['length'] / df['length'].max() if df['length'].max() > 0 else df['length']
    
    return df

if __name__ == "__main__":
    df = extract_features('data/traffic_raw.pcap')
    df.to_csv('data/traffic_features.csv', index=False)
    print("Features extracted and saved to traffic_features.csv")
