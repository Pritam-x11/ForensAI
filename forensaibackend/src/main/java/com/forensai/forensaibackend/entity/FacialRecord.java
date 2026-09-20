package com.forensai.forensaibackend.entity;

import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;

@Entity
public class FacialRecord {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    private String personName;

    private String caseReference;

    private String evidenceReference;

    private String embeddingReference;

    public FacialRecord() {
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public String getPersonName() {
        return personName;
    }

    public void setPersonName(String personName) {
        this.personName = personName;
    }

    public String getCaseReference() {
        return caseReference;
    }

    public void setCaseReference(String caseReference) {
        this.caseReference = caseReference;
    }

    public String getEvidenceReference() {
        return evidenceReference;
    }

    public void setEvidenceReference(String evidenceReference) {
        this.evidenceReference = evidenceReference;
    }

    public String getEmbeddingReference() {
        return embeddingReference;
    }

    public void setEmbeddingReference(String embeddingReference) {
        this.embeddingReference = embeddingReference;
    }
}