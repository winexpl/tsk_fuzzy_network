package com.example.layers.resources;

@FunctionalInterface
public
interface QFunction<T> {

    /**
     * Applies this function to the given arguments.
     *
     * @param t the first function argument
     * @param u the second function argument
     * @return the function result
     */
    T apply(T x, T sigma, T c, T b);
}