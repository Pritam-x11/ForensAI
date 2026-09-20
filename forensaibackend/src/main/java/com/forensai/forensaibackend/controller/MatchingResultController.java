package com.forensai.forensaibackend.controller;

import com.forensai.forensaibackend.entity.MatchingResult;
import com.forensai.forensaibackend.repository.MatchingResultRepository;

import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/matching-results")
public class MatchingResultController {

    private final MatchingResultRepository matchingResultRepository;

    public MatchingResultController(
            MatchingResultRepository matchingResultRepository) {

        this.matchingResultRepository = matchingResultRepository;
    }

    @GetMapping("/evidence/{evidenceId}")
    public List<MatchingResult> getResultsByEvidence(
            @PathVariable Long evidenceId) {

        return matchingResultRepository.findByEvidenceId(evidenceId);
    }
}