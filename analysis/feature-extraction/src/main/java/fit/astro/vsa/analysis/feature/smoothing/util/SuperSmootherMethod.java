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
package fit.astro.vsa.analysis.feature.smoothing.util;

/**
 *
 * @author Kyle.Johnston
 */
public abstract class SuperSmootherMethod {

    private SuperSmootherResults superSmootherResults;

    /**
     *
     * @return
     */
    abstract public SuperSmootherResults execute();

    /**
     * @return the superSmootherResults
     */
    public SuperSmootherResults getSuperSmootherResults() {
        return superSmootherResults;
    }

    /**
     * @param superSmootherResults the superSmootherResults to set
     */
    public void setSuperSmootherResults(SuperSmootherResults superSmootherResults) {
        this.superSmootherResults = superSmootherResults;
    }

}
