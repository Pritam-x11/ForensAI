package com.forensai.forensaibackend.repository;

import com.forensai.forensaibackend.entity.Evidence;
import org.springframework.data.jpa.repository.JpaRepository;

public interface EvidenceRepository extends JpaRepository<Evidence, Long> {
}