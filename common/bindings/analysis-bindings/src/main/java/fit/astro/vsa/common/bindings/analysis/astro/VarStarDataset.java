/*
 * Copyright (C) 2018 Kyle Johnston
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

import fit.astro.vsa.common.bindings.math.Real2DCurve;
import fit.astro.vsa.common.bindings.ml.input.MultiView;
import java.util.Map;

/**
 * Input variable dataset
 *
 * @author Kyle Johnston <kyjohnst2000@my.fit.edu>
 */
public class VarStarDataset {

//<editor-fold defaultstate="collapsed" desc="Variables">
    private final String description;
    private final Map<Integer, Real2DCurve> timeD;
    private final Map<Integer, VarStarInformation> info;
    
    private MultiView multiViewSideData;
//</editor-fold>

    /**
     * Input variable dataset
     *
     * @param description The origin of the dataset
     * @param timeD The time domain data (per entry)
     * @param info The associated meta data (per entry)
     */
    public VarStarDataset(String description,
            Map<Integer, Real2DCurve> timeD,
            Map<Integer, VarStarInformation> info) {
        this.description = description;
        this.timeD = timeD;
        this.info = info;
    }

//<editor-fold defaultstate="collapsed" desc="Getter and Setter">
    /**
     *
     * @return The origin of the dataset
     */
    public String getDescription() {
        return description;
    }

    /**
     *
     * @return The time domain data (per entry)
     */
    public Map<Integer, Real2DCurve> getTimeD() {
        return timeD;
    }

    /**
     *
     * @return The associated meta data (per entry)
     */
    public Map<Integer, VarStarInformation> getInfo() {
        return info;
    }

    /**
     *
     * @return
     */
    public MultiView getMultiViewSideData() {
        return multiViewSideData;
    }

    /**
     *
     * @param multiViewSideData
     */
    public void setMultiViewSideData(MultiView multiViewSideData) {
        this.multiViewSideData = multiViewSideData;
    }
//</editor-fold>

}
