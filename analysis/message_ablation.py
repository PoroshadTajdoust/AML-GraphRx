# Manuscript-linked analysis: Three paired seeds and three message-graph conditions
# Source lineage and packaging changes: provenance/source_lineage.json
"""R10B: genuine edge-message ablation of frozen R5G gene-first R-GAE.

Scientific boundaries: retrospective internal discovery-only. The decoder always
learns the ORIGINAL biological links; only encoder message passing changes.
No AML labels or BeatAML used in training; no independent transport claims.

Paste this entire file into one Colab cell (or run with %run /content/...).
Requirements: torch, torch-geometric, python-igraph, leidenalg, scipy, sklearn.
"""
from __future__ import annotations
import os as _release_os
try:
    from IPython.display import display
except ImportError:

    def display(value):
        print(value)
import sys, subprocess, json, hashlib, os, random, copy, warnings
from pathlib import Path
from io import BytesIO
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
for package, module in [('torch-geometric', 'torch_geometric'), ('python-igraph', 'igraph'), ('leidenalg', 'leidenalg')]:
    try:
        __import__(module)
    except ImportError:
        raise RuntimeError('Install the recorded analysis dependencies before running this script; see docs/ENVIRONMENT.md.')
from torch_geometric.nn import RGCNConv
from torch_geometric.utils import negative_sampling
import igraph as ig
import leidenalg
try:
    pass
    pass
except ImportError:
    pass
PROJECT = Path(_release_os.environ.get('AML_GRAPHRX_ROOT', '/content/drive/MyDrive/AML_GraphRx'))
AUDIT = PROJECT / 'results' / 'audit'
STEP5 = PROJECT / 'data' / 'patient_graphs' / '20260823T123040Z'
WORK = PROJECT / 'results' / 'r10b_graph_edge_ablation_v1'
AUDIT.mkdir(parents=True, exist_ok=True)
WORK.mkdir(parents=True, exist_ok=True)
RANK = STEP5 / '20260823T123040Z_patient_rank_raw_float64.npy'
TOPO = STEP5 / '20260823T123040Z_shared_locked_topology.pt'
DISC = AUDIT / '452_R5DICE_discovery_ledger.csv'
F481 = AUDIT / '481_R5G_predeclared_protocol.json'
F483 = AUDIT / '483_R5G_gene_embeddings.npy'
F486 = AUDIT / '486_R5G_patient_projection.npy'
F488 = AUDIT / '488_R5G_cross_cohort_MNN_edges.csv'
F489 = AUDIT / '489_R5G_Leiden_raw_assignments.csv'
F492 = AUDIT / '492_R5G_recurrent_state_patient_assignments.csv'
F497 = AUDIT / '497_R5G_manifest.csv'
F752 = AUDIT / '752_R10A_original_R5G_fidelity.csv'
F760 = AUDIT / '760_R10B_protocol.json'
F761 = AUDIT / '761_R10B_parent_integrity.csv'
F762 = AUDIT / '762_R10B_graph_rewiring_audit.csv'
F763 = AUDIT / '763_R10B_training_history.csv'
F764 = AUDIT / '764_R10B_arm_discovery_metrics.csv'
F765 = AUDIT / '765_R10B_paired_descriptive_contrasts.csv'
F766 = AUDIT / '766_R10B_final_summary.csv'
F767 = AUDIT / '767_R10B_manifest.csv'
SEEDS = [20261031, 20261101, 20261102]
ARMS = ['REAL_GRAPH', 'DEGREE_PRESERVING_REWIRE', 'NO_MESSAGE_PASSING']
REL = ['signaling', 'PPI', 'TF_regulation', 'kinase_substrate']
N_GENES = 9554
D = 32
N_PATIENTS = 1733
N_EPOCHS = 30
MAX_POS = 30000
LR = 0.0005
WEIGHT_DECAY = 1e-05
K = 10
N_BOOT = 30
EDGE_KEEP = 0.8
RES = 1.0
MIN_N = 40
MIN_COH = 4
MAX_COH_SHARE = 0.4
MIN_JAC = 0.6
SPLIT_SEED = 20261004
GSES = ['GSE15434', 'GSE161532', 'GSE1729', 'GSE34860', 'GSE37021', 'GSE37642', 'GSE6891', 'GSE71014']
GSE_COUNTS = [251, 61, 43, 78, 97, 562, 537, 104]
REWIRE_MULTIPLIER = 3
REWIRE_SEED = 20261029
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('R10B running on', DEVICE, 'torch', torch.__version__)
if DEVICE.type == 'cpu':
    print('WARNING: 9 x 30-epoch R-GAE fits may be slow on CPU; Colab T4 recommended.')

def require(cond, msg):
    if not bool(cond):
        raise RuntimeError(msg)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def write_once(path, payload):
    path = Path(path)
    if path.exists():
        require(path.read_bytes() == payload, 'Existing output differs, preserved: ' + str(path))
        print('PASS existing:', path.name)
        return
    temp = path.with_name(path.name + '.partial')
    temp.write_bytes(payload)
    os.replace(temp, path)
    print('CREATED:', path.name)

def save_json(path, obj):
    write_once(path, (json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + '\n').encode())

def save_csv(path, df, keys=None):
    df = df.copy()
    if keys:
        df = df.sort_values(keys, kind='mergesort').reset_index(drop=True)
    write_once(path, df.to_csv(index=False, lineterminator='\n', float_format='%.17g').encode())

def check_manifest():
    m = pd.read_csv(F497, dtype=str)
    require({'path', 'sha256'}.issubset(m.columns), 'R5G manifest schema changed')
    for p in [F481, F483, F486, F488, F489, F492, DISC, TOPO]:
        require(p.is_file(), 'Missing locked parent ' + str(p))
        sub = m.loc[m.path.map(lambda x: Path(str(x)).name) == p.name]
        require(len(sub) == 1, 'R5G manifest parent missing or ambiguous: ' + p.name)
        require(sub.sha256.iloc[0] == sha(p), 'R5G parent SHA mismatch: ' + p.name)
    require(RANK.is_file(), 'Locked expression rank matrix missing')
    require(F752.is_file(), 'Please run successful R10A first; missing 752 fidelity')
    ff = pd.read_csv(F752)
    require(str(ff['R5G_exact_reproduction_pass'].iloc[0]).lower() == 'true', 'R10A fidelity gate not passed')
    require(abs(float(ff['GNN_MNN_edge_jaccard'].iloc[0]) - 1) < 1e-12, 'R10A MNN fidelity failed')

def normalize_basis(e):
    e = np.asarray(e, dtype=np.float32).copy()
    require(e.shape == (N_GENES, D), 'Bad basis shape')
    e -= e.mean(axis=0, keepdims=True)
    sd = e.std(axis=0, keepdims=True)
    e /= np.where(sd < 1e-08, 1.0, sd)
    e /= np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)
    return e.astype(np.float32)

def patient_projection(X, gses, E):
    P = np.zeros((len(gses), D), dtype=np.float32)
    for g in GSES:
        ix = np.where(gses == g)[0]
        x = X[ix]
        med = np.median(x, axis=0, keepdims=True)
        p = (x - med) @ E / np.sqrt(N_GENES)
        m = np.median(p, axis=0, keepdims=True)
        q25 = np.quantile(p, 0.25, axis=0, keepdims=True)
        q75 = np.quantile(p, 0.75, axis=0, keepdims=True)
        scale = q75 - q25
        scale[scale < 1e-06] = 1.0
        p = np.clip((p - m) / scale, -8, 8)
        p /= np.maximum(np.linalg.norm(p, axis=1, keepdims=True), 1e-12)
        P[ix] = p.astype(np.float32)
    return P

def create_mnn(P, gses):
    edges = []
    for i, g1 in enumerate(GSES):
        for g2 in GSES[i + 1:]:
            a = np.where(gses == g1)[0]
            b = np.where(gses == g2)[0]
            sim = P[a] @ P[b].T
            ka = min(K, len(b))
            kb = min(K, len(a))
            ab = np.argpartition(-sim, kth=ka - 1, axis=1)[:, :ka]
            ba = np.argpartition(-sim.T, kth=kb - 1, axis=1)[:, :kb]
            back = [set(z.tolist()) for z in ba]
            for ai in range(len(a)):
                for bj in ab[ai]:
                    bj = int(bj)
                    if ai in back[bj] and sim[ai, bj] > 0:
                        edges.append((int(a[ai]), int(b[bj]), float(sim[ai, bj])))
    require(bool(edges), 'No positive cross-cohort MNN edges')
    return edges

def leiden_graph(n, edges, seed):
    g = ig.Graph(n=n, edges=[(int(a), int(b)) for a, b, w in edges], directed=False)
    g.es['weight'] = [float(w) for a, b, w in edges]
    return np.asarray(leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition, weights='weight', resolution_parameter=RES, seed=int(seed)).membership, dtype=int)

def state_metrics(labels, edges, gses):
    ids = np.unique(labels)
    members = {int(c): set(np.flatnonzero(labels == c).tolist()) for c in ids}
    endpoints = np.asarray([(a, b) for a, b, w in edges], dtype=int)
    weights = np.asarray([w for a, b, w in edges], dtype=float)
    n_keep = int(round(EDGE_KEEP * len(edges)))
    rng = np.random.default_rng(SPLIT_SEED + 9000)
    js = {int(c): [] for c in ids}
    for t in range(N_BOOT):
        idx = np.sort(rng.choice(len(edges), n_keep, replace=False))
        e = [(int(endpoints[j, 0]), int(endpoints[j, 1]), float(weights[j])) for j in idx]
        lab = leiden_graph(len(gses), e, SPLIT_SEED + t + 1)
        cms = [set(np.flatnonzero(lab == c).tolist()) for c in np.unique(lab)]
        for c, s in members.items():
            js[c].append(max((len(s & ss) / len(s | ss) for ss in cms)))
    median_j = {c: float(np.median(v)) for c, v in js.items()}
    passed = []
    for c, s in members.items():
        counts = pd.Series(gses[list(s)]).value_counts()
        if len(s) >= MIN_N and len(counts) >= MIN_COH and (float(counts.max() / len(s)) <= MAX_COH_SHARE) and (median_j[c] >= MIN_JAC):
            passed.append(c)
    valid = np.isin(labels, passed)
    nmi = float(normalized_mutual_info_score(gses[valid], labels[valid])) if valid.any() and len(set(labels[valid])) >= 2 else float('nan')
    return dict(states=len(passed), coverage=float(valid.mean()), stability=float(np.median([median_j[c] for c in passed])) if passed else float('nan'), cohort_nmi=nmi, labels=np.where(valid, labels, -1), mnn_coverage=len({i for a, b, w in edges for i in (a, b)}) / len(gses))

def rewire_relation_degree_preserving(edge, seed):
    """Directed double-edge swaps: keep src/out and dst/in degrees exactly.
       Preserve original self-loops, prohibit new self-loops/parallel edges.
       Does not guarantee connectedness or symmetric PPI."""
    e = edge.detach().cpu().numpy().astype(np.int64)
    loops = e[:, e[0] == e[1]]
    rows = e[0, e[0] != e[1]].copy()
    cols = e[1, e[0] != e[1]].copy()
    m = len(rows)
    keys = {(int(a), int(b)) for a, b in zip(rows, cols)}
    require(len(keys) == m, 'Relation contains duplicate nonself arcs; strict rewiring halts')
    rng = np.random.default_rng(seed)
    accepted = 0
    attempted = REWIRE_MULTIPLIER * m
    for _ in range(attempted):
        i, j = rng.integers(0, m, size=2)
        if i == j:
            continue
        a, b = (int(rows[i]), int(cols[i]))
        c, d = (int(rows[j]), int(cols[j]))
        if a == c or b == d or a == d or (c == b):
            continue
        if (a, d) in keys or (c, b) in keys:
            continue
        keys.remove((a, b))
        keys.remove((c, d))
        keys.add((a, d))
        keys.add((c, b))
        cols[i], cols[j] = (d, b)
        accepted += 1
    require(accepted > 0, 'Rewiring made zero accepted swaps')
    out = np.concatenate((np.stack([rows, cols]), loops), axis=1)
    require(out.shape == e.shape, 'Rewire arc count changed')
    src0 = np.bincount(e[0], minlength=N_GENES)
    src1 = np.bincount(out[0], minlength=N_GENES)
    dst0 = np.bincount(e[1], minlength=N_GENES)
    dst1 = np.bincount(out[1], minlength=N_GENES)
    require(np.array_equal(src0, src1) and np.array_equal(dst0, dst1), 'Rewire degree conservation FAILED')
    oldset = {(int(a), int(b)) for a, b in zip(e[0], e[1])}
    newset = {(int(a), int(b)) for a, b in zip(out[0], out[1])}
    overlap = len(oldset & newset) / len(oldset | newset)
    return (torch.from_numpy(out).long().contiguous(), dict(accepted_swaps=accepted, attempts=attempted, directed_edge_jaccard=overlap, n_edges=e.shape[1], n_selfloops=loops.shape[1]))

def build_message_graph(rel_edges, no_message=False):
    if no_message:
        return (torch.empty((2, 0), dtype=torch.long, device=DEVICE), torch.empty((0,), dtype=torch.long, device=DEVICE))
    edges = []
    types = []
    for r, e in enumerate(rel_edges):
        edges.append(e)
        types.append(torch.full((e.shape[1],), r, dtype=torch.long))
    ppi = rel_edges[1]
    rev = ppi[:, ppi[0] != ppi[1]].flip(0).contiguous()
    edges.append(rev)
    types.append(torch.full((rev.shape[1],), 1, dtype=torch.long))
    ei = torch.cat(edges, dim=1)
    et = torch.cat(types)
    ix = torch.argsort(et, stable=True)
    return (ei[:, ix].to(DEVICE), et[ix].to(DEVICE))

class GeneRGAE(nn.Module):

    def __init__(self):
        super().__init__()
        self.node_embedding = nn.Embedding(N_GENES, 32)
        self.input_projection = nn.Linear(32, 64)
        self.conv1 = RGCNConv(64, 64, num_relations=4, aggr='mean', root_weight=True, bias=True, is_sorted=True)
        self.conv2 = RGCNConv(64, 64, num_relations=4, aggr='mean', root_weight=True, bias=True, is_sorted=True)
        self.norm1 = nn.LayerNorm(64)
        self.norm2 = nn.LayerNorm(64)
        self.latent_head = nn.Linear(64, D)
        self.relation_embedding = nn.Parameter(torch.empty(4, D))
        nn.init.xavier_uniform_(self.relation_embedding)

    def encode(self, edge_index, edge_type):
        ids = torch.arange(N_GENES, device=DEVICE)
        h = F.gelu(self.input_projection(self.node_embedding(ids)))
        h = self.norm1(h + F.gelu(self.conv1(h, edge_index, edge_type)))
        h = self.norm2(h + F.gelu(self.conv2(h, edge_index, edge_type)))
        return self.latent_head(h)

    def score(self, z, e, r):
        return (z[e[0]] * self.relation_embedding[r].unsqueeze(0) * z[e[1]]).sum(dim=1)

def fixed_seed(seed):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def tensor_checkpoint(path, payload):
    temp = path.with_name(path.name + '.partial')
    torch.save(payload, temp)
    os.replace(temp, path)

def run_fit(arm, seed, mp_graph, truth_pos):
    """Checkpoint every 5 epochs. Same seed initializes every arm of a triplet.
       Original biological relation edges remain reconstruction targets."""
    arm_dir = WORK / f'{seed}_{arm}'
    arm_dir.mkdir(exist_ok=True, parents=True)
    checkpoint = arm_dir / 'resume.pt'
    embedding_path = arm_dir / 'gene_embedding32.npy'
    hist_path = arm_dir / 'training.csv'
    if embedding_path.exists() and hist_path.exists():
        e = np.load(embedding_path, allow_pickle=False)
        require(e.shape == (N_GENES, D), 'Bad cached embedding')
        require(len(pd.read_csv(hist_path)) == N_EPOCHS, 'Bad cached training history')
        print('RESUME completed:', seed, arm, flush=True)
        return (e, pd.read_csv(hist_path), embedding_path)
    fixed_seed(seed)
    model = GeneRGAE().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    start = 1
    history = []
    if checkpoint.exists():
        ck = torch.load(checkpoint, map_location=DEVICE, weights_only=False)
        require(ck['seed'] == seed and ck['arm'] == arm, 'Checkpoint mismatch')
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['optimizer'])
        start = int(ck['epoch']) + 1
        history = ck['history']
        print('RESUME partial:', seed, arm, 'epoch', start, flush=True)
    ei, et = mp_graph
    for epoch in range(start, N_EPOCHS + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        z = model.encode(ei, et)
        losses = []
        for r in range(4):
            pos_all = truth_pos[r]
            n_use = min(MAX_POS, pos_all.shape[1])
            gen = torch.Generator(device='cpu')
            gen.manual_seed(seed + epoch * 100 + r)
            if n_use < pos_all.shape[1]:
                pos = pos_all[:, torch.randperm(pos_all.shape[1], generator=gen)[:n_use]]
            else:
                pos = pos_all
            torch.manual_seed(seed + epoch * 1000 + r)
            neg = negative_sampling(edge_index=pos_all, num_nodes=N_GENES, num_neg_samples=pos.shape[1], method='sparse')
            pos = pos.to(DEVICE)
            neg = neg.to(DEVICE)
            scores_p = model.score(z, pos, r)
            scores_n = model.score(z, neg, r)
            l = (F.binary_cross_entropy_with_logits(scores_p, torch.ones_like(scores_p)) + F.binary_cross_entropy_with_logits(scores_n, torch.zeros_like(scores_n))) / 2
            losses.append(l)
        loss = torch.stack(losses).mean()
        require(bool(torch.isfinite(loss).item()), 'Nonfinite training loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        history.append(dict(seed=seed, arm=arm, epoch=epoch, mean_relation_BCE=float(loss.detach().cpu())))
        if epoch == 1 or epoch % 5 == 0 or epoch == N_EPOCHS:
            print(f'  {seed} {arm}: epoch {epoch:02d}/30 loss={float(loss.detach().cpu()):.5f}', flush=True)
        if epoch % 5 == 0 or epoch == N_EPOCHS:
            tensor_checkpoint(checkpoint, dict(seed=seed, arm=arm, epoch=epoch, model=model.state_dict(), optimizer=opt.state_dict(), history=history))
    model.eval()
    with torch.no_grad():
        emb = model.encode(ei, et).detach().cpu().numpy().astype(np.float32)
    require(emb.shape == (N_GENES, D) and np.isfinite(emb).all(), 'Invalid trained embeddings')
    with BytesIO() as buf:
        np.save(buf, emb, allow_pickle=False)
        write_once(embedding_path, buf.getvalue())
    save_csv(hist_path, pd.DataFrame(history), ['epoch'])
    del model, opt
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return (emb, pd.DataFrame(history), embedding_path)
check_manifest()
locked_protocol = json.loads(F481.read_text())
require(locked_protocol['gene_latent_dimension'] == D, 'Original R5G dimension mismatch')
require(locked_protocol['gene_GAE_epochs'] == N_EPOCHS, 'Original R5G epoch mismatch')
parents = [RANK, TOPO, DISC, F481, F483, F486, F488, F489, F492, F497, F752]
provenance = {p.name: sha(p) for p in parents}
protocol = dict(id='AML_GRAPHRX_R10B_GRAPH_EDGE_MESSAGE_ABLATION_V1', status='RETROSPECTIVE_EXPLORATORY_INTERNAL_DISCOVERY', question='Does relational graph message passing help recurrent discovery states?', arms=ARMS, seeds=SEEDS, epochs=N_EPOCHS, encoder='Original R5G GeneRGAE 2 RGCNConv layers; 32-64-32; DistMult', latent_dimensions=D, relations=REL, positive_target='FROZEN ORIGINAL BIOLOGICAL NONSELF EDGES IN ALL ARMS', negative_sampling='Same relation-specific original-link negative sampling in every arm', graph_interventions={'REAL_GRAPH': 'Original frozen relation topology', 'DEGREE_PRESERVING_REWIRE': 'Directed within-relation double-edge swaps, exact in/out degrees and selfloops', 'NO_MESSAGE_PASSING': 'Empty neighbor arcs, retains RGCN root transforms and trainable decoder'}, matched_comparison='Identical seeds, 30 epochs, optimizer, edge prediction targets and downstream R5G state discovery', downstream='R10A exact original patient projection, cross-cohort MNN K10, RB Leiden 1.0, 30x80percent edge perturbation', primary_descriptive_metrics=['recurrent_coverage', 'recurrent_states_n', 'median_recurrent_stability', 'state_cohort_NMI_descriptive'], no_drug_response=True, no_cohort_exclusion=True, no_graph_edits_to_locked_outputs=True, no_external_state_transport=True, no_causal_or_clinical_superiority_claim=True, parent_sha256=provenance)
save_json(F760, protocol)
save_csv(F761, pd.DataFrame([dict(file=p.name, sha256=sha(p), bytes=p.stat().st_size) for p in parents]), ['file'])
ledger = pd.read_csv(DISC)
require({'analysis_unit_index_0based', 'gse'}.issubset(ledger.columns), 'Discovery ledger columns changed')
require(len(ledger) == N_PATIENTS, 'Discovery count mismatch')
require(ledger['analysis_unit_index_0based'].is_unique, 'Duplicate analysis indices')
require(set(ledger.gse) == set(GSES), 'Discovery cohort set mismatch')
require([int((ledger.gse == g).sum()) for g in GSES] == GSE_COUNTS, 'Discovery cohort sizes mismatch')
ix = ledger['analysis_unit_index_0based'].to_numpy(int)
gses = ledger['gse'].astype(str).to_numpy()
mm = np.load(RANK, mmap_mode='r', allow_pickle=False)
require(mm.shape[1] == N_GENES and ix.max() < mm.shape[0], 'Raw rank matrix incompatible')
X = np.array(mm[ix, :], dtype=np.float32, copy=True)
require(X.shape == (N_PATIENTS, N_GENES) and np.isfinite(X).all(), 'Discovery matrix invalid')
E_locked = np.load(F483, allow_pickle=False)
P = patient_projection(X, gses, E_locked)
require(float(np.max(np.abs(P - np.load(F486, allow_pickle=False)))) < 0.0001, 'R5G projection fidelity failed')
e0 = create_mnn(P, gses)
edge_old = pd.read_csv(F488)
set0 = {(a, b) for a, b, w in e0}
set1 = {(int(a), int(b)) for a, b in zip(edge_old.patient_node_u, edge_old.patient_node_v)}
require(set0 == set1, 'R5G MNN exact reconstruction failed')
base_labels = leiden_graph(N_PATIENTS, e0, SPLIT_SEED)
labs_ref = pd.read_csv(F489).set_index('analysis_unit_index_0based').loc[ix, 'raw_community'].to_numpy(int)
require(adjusted_rand_score(labs_ref, base_labels) > 0.999999, 'R5G Leiden reproduction failed')
base_metrics = state_metrics(base_labels, e0, gses)
require(base_metrics['states'] == 6 and abs(base_metrics['coverage'] - 1720 / 1733) < 1e-10, 'R5G recurrent state exactness failed')
print('PASS: frozen original R5G reproduction (all 1733 cases, same 6 states).', flush=True)
topo = torch.load(TOPO, map_location='cpu', weights_only=False)
require(int(topo['node_count']) == N_GENES, 'Topology node count mismatch')
orig = [topo['edge_index_by_relation'][r].long().contiguous() for r in REL]
for r, e in enumerate(orig):
    require(e.ndim == 2 and e.shape[0] == 2 and (e.shape[1] > 0), 'Malformed relation: ' + REL[r])
    require(int(e.min()) >= 0 and int(e.max()) < N_GENES, 'Out-of-range relation: ' + REL[r])
truth = [e[:, e[0] != e[1]].contiguous() for e in orig]
real_graph = build_message_graph(orig)
empty_graph = build_message_graph(orig, no_message=True)
rewired = []
rewiring_rows = []
for r, e in zip(REL, orig):
    rw, record = rewire_relation_degree_preserving(e, REWIRE_SEED + REL.index(r))
    rewired.append(rw)
    rewiring_rows.append({'relation': r, **record})
rewire_graph = build_message_graph(rewired)
save_csv(F762, pd.DataFrame(rewiring_rows), ['relation'])
print('PASS: exact per-relation in/out degree preserved under rewiring.', flush=True)
results = []
training_frames = []
for seed in SEEDS:
    for arm in ARMS:
        print(f'FITTING {arm} seed={seed}', flush=True)
        g = {'REAL_GRAPH': real_graph, 'DEGREE_PRESERVING_REWIRE': rewire_graph, 'NO_MESSAGE_PASSING': empty_graph}[arm]
        emb, history, path = run_fit(arm, seed, g, truth)
        training_frames.append(history)
        proj = patient_projection(X, gses, emb)
        mnn = create_mnn(proj, gses)
        comm = leiden_graph(N_PATIENTS, mnn, SPLIT_SEED)
        met = state_metrics(comm, mnn, gses)
        ari = float(adjusted_rand_score(base_metrics['labels'], met['labels']))
        results.append(dict(seed=seed, arm=arm, embeddings_path=str(path), embedding_sha256=sha(path), mnn_edges_n=len(mnn), mnn_node_coverage=met['mnn_coverage'], recurrent_states_n=met['states'], recurrent_coverage=met['coverage'], median_recurrent_stability=met['stability'], state_cohort_NMI_descriptive=met['cohort_nmi'], ARI_vs_original_state_labels_descriptive=ari))
        print(f"  STATES={met['states']} COVERAGE={met['coverage']:.4f} STABILITY={met['stability']:.4f}", flush=True)
metrics = pd.DataFrame(results)
save_csv(F763, pd.concat(training_frames, ignore_index=True), ['seed', 'arm', 'epoch'])
save_csv(F764, metrics, ['seed', 'arm'])
contrasts = []
for seed in SEEDS:
    ss = metrics.loc[metrics.seed == seed].set_index('arm')
    real = ss.loc['REAL_GRAPH']
    for cmp in ('DEGREE_PRESERVING_REWIRE', 'NO_MESSAGE_PASSING'):
        other = ss.loc[cmp]
        contrasts.append(dict(seed=seed, comparison='REAL_GRAPH_minus_' + cmp, delta_recurrent_coverage=float(real.recurrent_coverage - other.recurrent_coverage), delta_recurrent_states=int(real.recurrent_states_n - other.recurrent_states_n), delta_median_recurrent_stability=float(real.median_recurrent_stability - other.median_recurrent_stability), delta_cohort_NMI=float(real.state_cohort_NMI_descriptive - other.state_cohort_NMI_descriptive)))
save_csv(F765, pd.DataFrame(contrasts), ['comparison', 'seed'])
summary = []
for arm, df in metrics.groupby('arm'):
    summary.append(dict(arm=arm, seeds=len(df), mean_coverage=float(df.recurrent_coverage.mean()), median_coverage=float(df.recurrent_coverage.median()), mean_recurrent_states=float(df.recurrent_states_n.mean()), median_recurrent_stability=float(df.median_recurrent_stability.median()), mean_cohort_NMI=float(df.state_cohort_NMI_descriptive.mean())))
summary = pd.DataFrame(summary)
summary['GRAPH_TOPOLOGY_SUPERIORITY_ESTABLISHED'] = False
summary['reason'] = 'Internal retrospective 3-seed ablation; no independent matched external transport test or causal graph-specific inference'
save_csv(F766, summary, ['arm'])
manifest_files = [F760, F761, F762, F763, F764, F765, F766]
manifest_files += [WORK / f'{seed}_{arm}' / 'gene_embedding32.npy' for seed in SEEDS for arm in ARMS]
save_csv(F767, pd.DataFrame([dict(path=str(p), sha256=sha(p), bytes=p.stat().st_size) for p in manifest_files]), ['path'])
print('\n' + '=' * 105)
print('R10B COMPLETE — GRAPH-EDGE MESSAGE ABLATION (INTERNAL DISCOVERY ONLY)')
print('=' * 105)
print('\nALL ARMS:\n' + metrics.to_string(index=False))
print('\nPAIRED CONTRASTS:\n' + pd.DataFrame(contrasts).to_string(index=False))
print('\nFAMILY SUMMARY:\n' + summary.to_string(index=False))
print('\nIMPORTANT: No independent transport superiority is established by internal clustering metrics.')
print('OUTPUT DIR:', AUDIT)
