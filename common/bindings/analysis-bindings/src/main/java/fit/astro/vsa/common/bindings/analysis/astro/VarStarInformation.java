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
package fit.astro.vsa.common.bindings.analysis.astro;

import java.util.Map;

/**
 *
 * @author Kyle Johnston <kyjohnst2000@my.fit.edu>
 */
public class VarStarInformation {

    private final String id;
    private Map<Astrometrics, Double> astrometricValues;

    /**
     * Must have a name, might have other things
     *
     * @param id
     */
    public VarStarInformation(String id) {
        this.id = id;
    }

    public String getId() {
        return id;
    }

    public Map<Astrometrics, Double> getAstrometricValues() {
        return astrometricValues;
    }

    public void setAstrometricValues(Map<Astrometrics, Double> astrometricValues) {
        this.astrometricValues = astrometricValues;
    }

}
