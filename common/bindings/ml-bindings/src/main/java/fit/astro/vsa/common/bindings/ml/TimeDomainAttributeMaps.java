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
package fit.astro.vsa.common.bindings.ml;

import fit.astro.vsa.common.bindings.math.Real2DCurve;
import java.util.Map;

/**
 * The pairing of time domain measurements and classes that define the
 * underlying observation
 *
 * @author Kyle Johnston <kyjohnst2000@my.fit.edu>
 */
public class TimeDomainAttributeMaps {

    private final Map<Integer, String> setOfClasses;
    private final Map<Integer, Real2DCurve> setOfWaveforms;

    public TimeDomainAttributeMaps(
            Map<Integer, String> setOfClasses,
            Map<Integer, Real2DCurve> setOfWaveforms) {

        this.setOfClasses = setOfClasses;
        this.setOfWaveforms = setOfWaveforms;
    }

    /**
     * The set of labels that describe the data
     *
     * @return
     */
    public Map<Integer, String> getSetOfClasses() {
        return setOfClasses;
    }

    /**
     * The set of waveform data (collection of weighted points)
     *
     * @return
     */
    public Map<Integer, Real2DCurve> getSetOfWaveforms() {
        return setOfWaveforms;
    }

}
