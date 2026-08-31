from captionlm.eval_dataset import build_filing_url


def test_build_filing_url():
    url = build_filing_url("0000320193", "0001-21-000002", "pressrelease.htm")
    assert url == "https://www.sec.gov/Archives/edgar/data/320193/000121000002/pressrelease.htm"
