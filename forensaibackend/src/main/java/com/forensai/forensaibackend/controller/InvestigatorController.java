package com.forensai.forensaibackend.controller;

import com.forensai.forensaibackend.service.AiServiceClient;

import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/investigator")
public class InvestigatorController {

    private final AiServiceClient aiServiceClient;

    public InvestigatorController(
            AiServiceClient aiServiceClient) {

        this.aiServiceClient = aiServiceClient;
    }

    @PostMapping("/feedback")
    public String submitFeedback(
            @RequestParam("candidate_index") Long candidateIndex,
            @RequestParam("feedback") String feedback) {

        return aiServiceClient.requestRefinedSearch(
                candidateIndex,
                feedback
        );
    }

    @PostMapping("/refined-search")
    public String refinedSearch(
            @RequestBody String requestBody) {

        return aiServiceClient.performRefinedSearch(
                requestBody
        );
    }
}