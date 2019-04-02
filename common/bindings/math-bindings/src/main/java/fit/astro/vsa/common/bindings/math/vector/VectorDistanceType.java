/*
 * Copyright (C) 2016 Kyle Johnston
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package fit.astro.vsa.common.bindings.math.vector;

/**
 *
 * @author Kyle.Johnston
 */
public enum VectorDistanceType {

    /**
     * Specifies the "City Block"
     */
    CITY_BLOCK("City Block"),
    /**
     * Specifies the "Euclidean"
     */
    EUCLIDEAN_DISTANCE("Euclidean"),
    /**
     * Specifies the "Pearson"
     */
    PEARSON_DISTANCE("Pearson"),
    /**
     * Specifies the "Canberra"
     */
    CANBERRA_DISTANCE("Canberra");


    private final String methodLabel;

    /**
     * 
     * @param methodLabel 
     */
    VectorDistanceType(String methodLabel) {
        this.methodLabel = methodLabel;
    }

    /**
     * @return the classLabel
     */
    public String getMethodLabel() {
        return methodLabel;
    }
}
