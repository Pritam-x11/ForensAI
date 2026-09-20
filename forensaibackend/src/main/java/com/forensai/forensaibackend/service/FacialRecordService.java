package com.forensai.forensaibackend.service;

import com.forensai.forensaibackend.entity.FacialRecord;
import com.forensai.forensaibackend.repository.FacialRecordRepository;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Optional;

@Service
public class FacialRecordService {

    private final FacialRecordRepository facialRecordRepository;

    public FacialRecordService(FacialRecordRepository facialRecordRepository) {
        this.facialRecordRepository = facialRecordRepository;
    }

    public FacialRecord saveRecord(FacialRecord record) {
        return facialRecordRepository.save(record);
    }

    public List<FacialRecord> getAllRecords() {
        return facialRecordRepository.findAll();
    }

    public Optional<FacialRecord> getRecordById(Long id) {
        return facialRecordRepository.findById(id);
    }
}