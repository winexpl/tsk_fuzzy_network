package com.example.models.resources;

import lombok.AllArgsConstructor;
import lombok.Getter;

@AllArgsConstructor
public enum SpeciesOfIris {
    SETOSA(0), VERSICOLOR(1), VIRGINICA(2);
    @Getter
    private final int i;
}
