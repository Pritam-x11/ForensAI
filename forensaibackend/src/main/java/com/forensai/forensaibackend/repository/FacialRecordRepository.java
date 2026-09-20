package com.forensai.forensaibackend.repository;

import com.forensai.forensaibackend.entity.FacialRecord;
import org.springframework.data.jpa.repository.JpaRepository;

public interface FacialRecordRepository extends JpaRepository<FacialRecord, Long> {
}