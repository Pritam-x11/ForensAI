package com.forensai.forensaibackend.controller;

import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class ProtectedController {

  @GetMapping("/api/protected")
@PreAuthorize("hasRole('OFFICER')")
public String protectedEndpoint() {
    return "JWT Authentication + OFFICER Role is working!";
}
}