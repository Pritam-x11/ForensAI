package com.forensai.forensaibackend.controller;

import com.forensai.forensaibackend.entity.FacialRecord;
import com.forensai.forensaibackend.service.FacialRecordService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/facial-records")
public class FacialRecordController {

    private final FacialRecordService facialRecordService;

    public FacialRecordController(FacialRecordService facialRecordService) {
        this.facialRecordService = facialRecordService;
    }

    @PostMapping
    public ResponseEntity<FacialRecord> createRecord(
            @RequestBody FacialRecord record) {

        return ResponseEntity.ok(
                facialRecordService.saveRecord(record)
        );
    }

    @GetMapping
    public ResponseEntity<List<FacialRecord>> getAllRecords() {

        return ResponseEntity.ok(
                facialRecordService.getAllRecords()
        );
    }

    @GetMapping("/{id}")
    public ResponseEntity<FacialRecord> getRecordById(
            @PathVariable Long id) {

        return facialRecordService.getRecordById(id)
                .map(ResponseEntity::ok)
                .orElse(ResponseEntity.notFound().build());
    }
}