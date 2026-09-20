package com.forensai.forensaibackend.repository;

import com.forensai.forensaibackend.entity.MatchingResult;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;

public interface MatchingResultRepository
                extends JpaRepository<MatchingResult, Long> {

        List<MatchingResult> findByEvidenceId(Long evidenceId);
}