"""SessionData file lookup and cluster assignment, on small synthetic sessions in both layouts:
the older one (kilosorted/) and the cbm one (one <region>-ks/ folder per brain region)."""
import numpy as np
import pandas as pd
import pytest

from sldata import SessionData


def write_events(folder, timestamps=True):
    folder.mkdir(parents=True, exist_ok=True)
    n = 100
    pd.DataFrame(dict(timestamp_ms=np.arange(n) * 33.3 if timestamps else np.nan, bonsai_ts=np.arange(n) * 33,
                      nose_x=np.linspace(0, 800, n), nose_y=np.linspace(0, 1900, n),
                      bonsai_centroid_x=0, bonsai_centroid_y=0)).to_csv(folder / 'events.csv', index=False)


def write_ks(folder, clusters, templates_of_spikes):
    folder.mkdir(parents=True, exist_ok=True)
    n = len(clusters)
    np.save(folder / 'spike_times.npy', (np.arange(n) * 30).astype(np.uint64)[:, None])
    np.save(folder / 'spike_clusters.npy', np.asarray(clusters))
    np.save(folder / 'spike_templates.npy', np.asarray(templates_of_spikes))
    np.save(folder / 'templates.npy', np.ones((3, 5, 4)))
    pd.DataFrame(dict(cluster_id=[0, 1, 2], KSLabel=['good', 'mua', 'good'])).to_csv(
        folder / 'cluster_KSLabel.tsv', sep='\t', index=False)


@pytest.fixture
def cbm_root(tmp_path):
    exp = tmp_path / 'cbm-odor'
    write_events(exp / 'bonsai' / '9005' / 'x5', timestamps=False)  # before ephys alignment
    write_events(exp / 'events' / '9005' / 'x5')
    # phy merged clusters 0 and 2 into 3 in hc; ob is uncurated
    write_ks(exp / 'hc-ks' / '9005' / 'x5', clusters=[3, 1, 3, 1, 3, 1], templates_of_spikes=[0, 1, 2, 1, 0, 1])
    pd.DataFrame(dict(cluster_id=[3], group=['good'])).to_csv(
        exp / 'hc-ks' / '9005' / 'x5' / 'cluster_group.tsv', sep='\t', index=False)
    write_ks(exp / 'ob-ks' / '9005' / 'x5', clusters=[0, 0, 1, 2], templates_of_spikes=[0, 0, 1, 2])
    return tmp_path


def load(root, experiment='cbm-odor', **kw):
    return SessionData('9005', 'x5', experiment, base_path=str(root), min_spikes=1, verbose=False, **kw)


def test_cbm_layout_needs_a_region(cbm_root):
    with pytest.raises(ValueError, match='region'):
        load(cbm_root)


def test_cbm_layout_per_region(cbm_root):
    hc, ob = load(cbm_root, region='hc'), load(cbm_root, region='ob')
    assert sorted(c['cluster_id'] for c in hc.clusters.values()) == [1, 3]  # phy's merge, not templates
    assert {c['cluster_id']: c['label'] for c in hc.clusters.values()} == {1: 'mua', 3: 'good'}
    assert sorted(c['cluster_id'] for c in ob.clusters.values()) == [0, 1, 2]
    assert {c['region'] for c in hc.clusters.values()} == {'hc'}
    n_spikes = {c['cluster_id']: c['n_spikes'] for c in hc.clusters.values()}
    assert n_spikes == {1: 3, 3: 3}


def test_ephys_aligned_events_are_used(cbm_root):
    hc = load(cbm_root, region='hc')
    assert hc.events['timestamp_ms'].notna().all()
    assert (hc.position_x, hc.position_y) == ('nose_x', 'nose_y')
    assert hc.video_path is None  # no video in this session: no crash


def test_older_layout(tmp_path):
    exp = tmp_path / 'xclickbait-motivate'
    write_events(exp / 'events' / '9005' / 'x5')
    write_ks(exp / 'kilosorted' / '9005' / 'x5', clusters=[0, 1, 1, 2], templates_of_spikes=[0, 1, 1, 2])
    s = load(tmp_path, experiment='xclickbait-motivate')
    assert sorted(c['cluster_id'] for c in s.clusters.values()) == [0, 1, 2]
    assert s.clusters[1]['n_spikes'] == 2
