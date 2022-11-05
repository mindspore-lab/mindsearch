def compute_metrics(qids_to_relevant_passageids, qids_to_ranked_candidate_passages):
    """Compute MRR metric
    Args:
    p_qids_to_relevant_passageids (dict): dictionary of query-passage mapping
        Dict as read in with load_reference or load_reference_from_stream
    p_qids_to_ranked_candidate_passages (dict): dictionary of query-passage candidates
    Returns:
        dict: dictionary of metrics {'MRR': <MRR Score>}
    """
    all_scores = []
    for MaxMRRRank in [1, 10, 50, 100, 500, 1000]:
        MRR = 0
        ACC = 0
        Recall = 0
        ranking = []
        for qid in qids_to_ranked_candidate_passages:
            if qid in qids_to_relevant_passageids:
                ranking.append(0)
                target_pid = qids_to_relevant_passageids[qid]
                candidate_pid = qids_to_ranked_candidate_passages[qid]
                for i in range(0, MaxMRRRank):
                    if candidate_pid[i] in target_pid:
                        ACC += 1
                        MRR += 1 / (i + 1)
                        ranking.pop()
                        ranking.append(i + 1)
                        break
                Recall += (len(set(target_pid) & set(candidate_pid[:MaxMRRRank])) / len(target_pid))
        if len(ranking) == 0:
            raise IOError("No matching QIDs found. Are you sure you are scoring the evaluation set?")
        
        MRR = MRR / len(qids_to_relevant_passageids)
        Recall = Recall / len(qids_to_relevant_passageids)
        all_scores.append(('MRR @{}'.format(MaxMRRRank), MRR))
        all_scores.append(('Recall @{}'.format(MaxMRRRank), Recall))
    
    all_scores.append(('QueriesRanked', len(qids_to_ranked_candidate_passages)))
    return all_scores
