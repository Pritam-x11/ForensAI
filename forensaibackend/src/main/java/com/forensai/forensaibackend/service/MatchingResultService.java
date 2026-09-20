package com.forensai.forensaibackend.service;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.forensai.forensaibackend.entity.MatchingResult;
import com.forensai.forensaibackend.repository.MatchingResultRepository;
import org.springframework.stereotype.Service;

@Service
public class MatchingResultService {

        private final MatchingResultRepository matchingResultRepository;
        private final ObjectMapper objectMapper;

        public MatchingResultService(
                        MatchingResultRepository matchingResultRepository) {

                this.matchingResultRepository = matchingResultRepository;
                this.objectMapper = new ObjectMapper();
        }

        /*
         * PHOTO RESULTS
         */

        public void savePhotoResults(
                        Long evidenceId,
                        String aiResult) throws Exception {

                JsonNode root = objectMapper.readTree(aiResult);

                JsonNode faces = root.get("faces");

                if (faces == null || !faces.isArray()) {
                        return;
                }

                for (JsonNode face : faces) {

                        JsonNode qualityScoreNode =
                                        face.get("quality_score");

                        double qualityScore =
                                        qualityScoreNode != null
                                                        ? qualityScoreNode.asDouble()
                                                        : 0.0;

                        JsonNode explainableResult =
                                        face.get("explainable_result");

                        if (explainableResult == null) {
                                continue;
                        }

                        JsonNode candidates =
                                        explainableResult.get("candidates");

                        if (candidates == null || !candidates.isArray()) {
                                continue;
                        }

                        for (JsonNode candidate : candidates) {

                                MatchingResult result =
                                                new MatchingResult();

                                result.setEvidenceId(evidenceId);

                                JsonNode candidateIndex =
                                                candidate.get("candidate_index");

                                if (candidateIndex != null) {
                                        result.setCandidateId(
                                                        candidateIndex.asLong());
                                }

                                JsonNode filename =
                                                candidate.get("filename");

                                if (filename != null) {
                                        result.setCandidateName(
                                                        filename.asText());
                                }

                                JsonNode distance =
                                                candidate.get("distance");

                                if (distance != null) {
                                        result.setDistance(
                                                        distance.asDouble());
                                }

                                JsonNode similarity =
                                                candidate.get("similarity_score");

                                if (similarity != null) {
                                        result.setSimilarityScore(
                                                        similarity.asDouble());
                                }

                                result.setQualityScore(
                                                qualityScore);

                                JsonNode finalScore =
                                                candidate.get("final_score");

                                if (finalScore != null) {
                                        result.setFinalScore(
                                                        finalScore.asDouble());
                                }

                                matchingResultRepository.save(result);
                        }
                }
        }

        /*
         * VIDEO RESULTS
         */

        public void saveVideoResults(
                        Long evidenceId,
                        String aiResult) throws Exception {

                JsonNode root =
                                objectMapper.readTree(aiResult);

                JsonNode trackCandidates =
                                root.get("track_candidates");

                if (trackCandidates == null
                                || !trackCandidates.isObject()) {
                        return;
                }

                JsonNode trackQuality =
                                root.get("track_quality");

                trackCandidates.fields().forEachRemaining(
                                trackEntry -> {

                                        String trackId =
                                                        trackEntry.getKey();

                                        JsonNode candidates =
                                                        trackEntry.getValue();

                                        double qualityScore = 0.0;

                                        if (trackQuality != null
                                                        && trackQuality.has(trackId)) {

                                                JsonNode qualityNode =
                                                                trackQuality
                                                                                .get(trackId)
                                                                                .get("quality_score");

                                                if (qualityNode != null) {
                                                        qualityScore =
                                                                        qualityNode.asDouble();
                                                }
                                        }

                                        if (candidates == null
                                                        || !candidates.isArray()) {
                                                return;
                                        }

                                        for (JsonNode candidate : candidates) {

                                                MatchingResult result =
                                                                new MatchingResult();

                                                result.setEvidenceId(
                                                                evidenceId);

                                                JsonNode candidateIndex =
                                                                candidate.get("candidate_index");

                                                if (candidateIndex != null) {
                                                        result.setCandidateId(
                                                                        candidateIndex.asLong());
                                                }

                                                JsonNode filename =
                                                                candidate.get("filename");

                                                if (filename != null) {
                                                        result.setCandidateName(
                                                                        filename.asText());
                                                }

                                                JsonNode distance =
                                                                candidate.get("distance");

                                                if (distance != null) {
                                                        result.setDistance(
                                                                        distance.asDouble());
                                                }

                                                JsonNode similarity =
                                                                candidate.get("similarity_score");

                                                if (similarity != null) {
                                                        result.setSimilarityScore(
                                                                        similarity.asDouble());
                                                }

                                                result.setQualityScore(
                                                                qualityScore);

                                                JsonNode finalScore =
                                                                candidate.get("final_score");

                                                if (finalScore != null) {
                                                        result.setFinalScore(
                                                                        finalScore.asDouble());
                                                }

                                                matchingResultRepository.save(
                                                                result);
                                        }
                                });
        }
}