package com.forensai.forensaibackend.controller;

import com.forensai.forensaibackend.entity.Evidence;
import com.forensai.forensaibackend.repository.EvidenceRepository;
import com.forensai.forensaibackend.service.AiServiceClient;
import com.forensai.forensaibackend.service.MatchingResultService;

import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.UUID;

@RestController
@RequestMapping("/api/evidence")
public class EvidenceController {

    private final EvidenceRepository evidenceRepository;
    private final AiServiceClient aiServiceClient;
    private final MatchingResultService matchingResultService;

    private final Path uploadDirectory = Paths.get("uploads");

    public EvidenceController(
            EvidenceRepository evidenceRepository,
            AiServiceClient aiServiceClient,
            MatchingResultService matchingResultService) {

        this.evidenceRepository = evidenceRepository;
        this.aiServiceClient = aiServiceClient;
        this.matchingResultService = matchingResultService;
    }

    @PostMapping("/upload")
    public Evidence uploadEvidence(
            @RequestParam("file") MultipartFile file,
            @RequestParam("caseReference") String caseReference,
            @RequestParam("evidenceType") String evidenceType)
            throws Exception {

        if (file.isEmpty()) {
            throw new IllegalArgumentException("File cannot be empty");
        }

        Files.createDirectories(uploadDirectory);

        String originalFileName =
                file.getOriginalFilename();

        if (originalFileName == null ||
                originalFileName.isBlank()) {

            throw new IllegalArgumentException(
                    "Invalid file name");
        }

        String safeFileName =
                Paths.get(originalFileName)
                        .getFileName()
                        .toString();

        String storedFileName =
                UUID.randomUUID() + "_" + safeFileName;

        Path filePath =
                uploadDirectory.resolve(storedFileName);

        byte[] fileBytes = file.getBytes();

        Files.write(filePath, fileBytes);

        Evidence evidence = new Evidence();

        evidence.setCaseReference(caseReference);
        evidence.setEvidenceType(
                evidenceType.toUpperCase());
        evidence.setFileName(safeFileName);
        evidence.setFilePath(filePath.toString());
        evidence.setProcessingStatus("UPLOADED");

        Evidence savedEvidence =
                evidenceRepository.save(evidence);

        /*
         * PHOTO PROCESSING
         */

        if ("PHOTO".equalsIgnoreCase(evidenceType)) {

            String aiResult =
                    aiServiceClient.processPhoto(
                            fileBytes,
                            safeFileName);

            System.out.println(
                    "AI PHOTO RESULT:");

            System.out.println(aiResult);

            matchingResultService.savePhotoResults(
                    savedEvidence.getId(),
                    aiResult);
        }

        /*
         * VIDEO PROCESSING
         */

        if ("VIDEO".equalsIgnoreCase(evidenceType)) {

            String aiResult =
                    aiServiceClient.processVideo(
                            fileBytes,
                            safeFileName);

            System.out.println(
                    "AI VIDEO RESULT:");

            System.out.println(aiResult);

            matchingResultService.saveVideoResults(
                    savedEvidence.getId(),
                    aiResult);
        }

        return savedEvidence;
    }

    @GetMapping
    public List<Evidence> getAllEvidence() {
        return evidenceRepository.findAll();
    }
}