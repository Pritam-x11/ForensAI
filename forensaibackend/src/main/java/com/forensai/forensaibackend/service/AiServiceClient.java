package com.forensai.forensaibackend.service;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.client.RestClient;

@Service
public class AiServiceClient {

    private final RestClient restClient;

    public AiServiceClient(
            @Value("${ai.service.url}") String aiServiceUrl) {

        this.restClient = RestClient.builder()
                .baseUrl(aiServiceUrl)
                .build();
    }

    public String processPhoto(
            byte[] fileBytes,
            String fileName) {

        ByteArrayResource fileResource =
                new ByteArrayResource(fileBytes) {

                    @Override
                    public String getFilename() {
                        return fileName;
                    }
                };

        MultiValueMap<String, Object> body =
                new LinkedMultiValueMap<>();

        body.add("file", fileResource);

        return restClient.post()
                .uri("/api/photo/upload")
                .contentType(MediaType.MULTIPART_FORM_DATA)
                .body(body)
                .retrieve()
                .body(String.class);
    }

    public String processVideo(
            byte[] fileBytes,
            String fileName) {

        ByteArrayResource fileResource =
                new ByteArrayResource(fileBytes) {

                    @Override
                    public String getFilename() {
                        return fileName;
                    }
                };

        MultiValueMap<String, Object> body =
                new LinkedMultiValueMap<>();

        body.add("file", fileResource);

        return restClient.post()
                .uri("/api/video/upload")
                .contentType(MediaType.MULTIPART_FORM_DATA)
                .body(body)
                .retrieve()
                .body(String.class);
    }

    public String submitInvestigatorFeedback(
            Long candidateIndex,
            String feedback) {

        return restClient.post()
                .uri(uriBuilder -> uriBuilder
                        .path("/api/investigator/feedback")
                        .queryParam("candidate_index", candidateIndex)
                        .queryParam("feedback", feedback)
                        .build())
                .retrieve()
                .body(String.class);
    }

    public String requestRefinedSearch(
            Long candidateIndex,
            String feedback) {

        return restClient.post()
                .uri(uriBuilder -> uriBuilder
                        .path("/api/investigator/feedback")
                        .queryParam("candidate_index", candidateIndex)
                        .queryParam("feedback", feedback)
                        .build())
                .retrieve()
                .body(String.class);
    }

    public String performRefinedSearch(
            String requestBody) {

        return restClient.post()
                .uri("/api/investigator/refined-search")
                .contentType(MediaType.APPLICATION_JSON)
                .body(requestBody)
                .retrieve()
                .body(String.class);
    }
}