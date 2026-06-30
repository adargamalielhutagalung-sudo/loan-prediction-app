import streamlit as st
import joblib
import numpy as np

# ====================================================================
# 1. KONFIGURASI HALAMAN UTAMA STREAMLIT
# ====================================================================
st.set_page_config(
    page_title="Deployment Aplikasi Loan Prediction",
    page_icon="",
    layout="centered"
)

st.title("Aplikasi Web Analisis Kelayakan Kredit Pinjaman")


# ====================================================================
# 2. FUNGSI LOAD SCALER & MODEL MENGGUNAKAN JOBLIB
# ====================================================================
@st.cache_resource
def load_ml_resources():
    scaler = joblib.load('scaler.pkl')
    model = joblib.load('loan_prediction_model.pkl')
    return scaler, model

scaler, model = None, None
try:
    scaler, model = load_ml_resources()

except FileNotFoundError:
    st.error(" Berkas .pkl tidak ditemukan! Pastikan file 'scaler.pkl' dan 'loan_prediction_model.pkl' berada di folder utama aplikasi.")
except Exception as e:
    st.error(f" Gagal memuat berkas biner: {e}")

# ====================================================================
# 3. FORM INPUT ANTARMUKA PENGGUNA (FORMAT INTEGER DENGAN SEPARATOR TITIK)
# ====================================================================
st.subheader(" Input Parameter Pengajuan Kredit")
st.write("Silakan isi formulir karakteristik finansial pemohon di bawah ini:")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Jenis Kelamin (Gender):", options=["Male", "Female"])
    married = st.selectbox("Status Pernikahan (Married):", options=["Yes", "No"])
    dependents = st.selectbox("Jumlah Tanggungan (Dependents):", options=["0", "1", "2", "3+"])
    education = st.selectbox("Tingkat Pendidikan (Education):", options=["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Wiraswasta/Pekerja Mandiri:", options=["No", "Yes"])
    property_area = st.selectbox("Area Properti Tempat Tinggal:", options=["Semi-Perkotaan (Semiurban)", "Perkotaan (Urban)", "Pedesaan (Rural)"])

with col2:
    # Menggunakan parameter value dengan format integer agar Streamlit otomatis memberikan pemisah titik (.)
    applicant_income = st.number_input("Pendapatan Bulanan Pemohon Utama (Rp):", min_value=0, value=5000000, step=500000)
    coapplicant_income = st.number_input("Pendapatan Bulanan Penjamin/Pasangan (Rp):", min_value=0, value=0, step=500000)
    loan_amount = st.number_input("Jumlah Nilai Pinjaman Yang Diajukan (Rp):", min_value=0, value=150000000, step=5000000)
    loan_term = st.number_input("Durasi Waktu Pinjaman (Tenor/Hari):", min_value=0, value=360, step=12)
    credit_history = st.selectbox("Catatan Kelayakan Kredit Sebelumnya:", options=["Memenuhi Syarat (1.0)", "Tidak Memenuhi Syarat (0.0)"])

# ====================================================================
# 4. PROSES ENCODING DATA (KATEGORIKAL -> NUMERIK MATRIKS ASLI)
# ====================================================================
gender_enc = 1 if gender == "Male" else 0
married_enc = 1 if married == "Yes" else 0
dependents_enc = 3 if dependents == "3+" else int(dependents)
edu_enc = 0 if education == "Graduate" else 1
emp_enc = 1 if self_employed == "Yes" else 0

if "Pedesaan" in property_area: prop_enc = 0
elif "Semi-Perkotaan" in property_area: prop_enc = 1
else: prop_enc = 2

cred_enc = 1.0 if "Memenuhi Syarat" in credit_history else 0.0

# ====================================================================
# 5. LOGIKA PREDIKSI & VALIDASI ATURAN KELAYAKAN DINAMIS
# ====================================================================
st.markdown("---")

if st.button(" Hitung Analisis Kelayakan Pinjaman", use_container_width=True):
    if scaler is not None and model is not None:
        try:
            # Penyetaraan matematis nilai rupiah ke skala matriks dasar model pkl kelompok
            income_scaled = applicant_income / 15000
            co_income_scaled = coapplicant_income / 15000
            loan_scaled = loan_amount / 15000000

            fitur_array = np.array([[
                gender_enc, married_enc, dependents_enc, edu_enc, emp_enc,
                income_scaled, co_income_scaled, loan_scaled, loan_term,
                cred_enc, prop_enc
            ]])
            
            fitur_tertransformasi = scaler.transform(fitur_array)
            
            # Mendapatkan probabilitas matematika asli berdasarkan algoritma pengklasifikasi kelompok
            probabilitas_raw = model.predict_proba(fitur_tertransformasi)[0]
            
            # Hitung total kapasitas finansial riil bulanan pemohon
            total_income_bulanan = applicant_income + coapplicant_income
            
            # Menghitung perkiraan cicilan dasar (Asumsi tenor dikonversi ke bulan: Hari / 30)
            tenor_bulan = max(loan_term / 30, 1)
            kemampuan_bayar_total = total_income_bulanan * tenor_bulan

            # ----------------------------------------------------------------
            # EVALUASI LOGIKA ATURAN BISNIS KREDIT (DINAMIS & SEIMBANG)
            # ----------------------------------------------------------------
            # Aturan 1: Jika catatan kredit buruk, otomatis ditolak mutlak
            if cred_enc == 0.0:
                prediksi_akhir = 0
                persen_ditolak = max(probabilitas_raw[0] * 100, 95.0)
                persen_disetujui = 100.0 - persen_ditolak
            
            # Aturan 2: Jika total pendapatan selama tenor tidak cukup melunasi pinjaman ( mustahil bayar )
            elif kemampuan_bayar_total < loan_amount:
                prediksi_akhir = 0
                persen_ditolak = max(probabilitas_raw[0] * 100, 85.50)
                persen_disetujui = 100.0 - persen_ditolak
                
            # Aturan 3: Jika lolos penyaringan dasar keuangan, gunakan keputusan murni dari probabilitas model ML kelompok
            else:
                # Menggunakan indeks keputusan berdasarkan probabilitas tertinggi
                prediksi_akhir = np.argmax(probabilitas_raw)
                persen_ditolak = probabilitas_raw[0] * 100
                persen_disetujui = probabilitas_raw[1] * 100
            # ----------------------------------------------------------------
            
            # 3. Menampilkan Hasil Akhir ke Layar Browser
            st.subheader(" Hasil Keputusan Sistem")
            if prediksi_akhir == 1:
                st.success(" **REKOMENDASI SISTEM: PENGAJUAN PINJAMAN LAYAK DISETUJUI (APPROVED)**")
            else:
                st.error(" **REKOMENDASI SISTEM: PENGAJUAN PINJAMAN BELUM LAYAK (REJECTED)**")
                if cred_enc == 0.0:
                    st.warning("Alasan Penolakan: Riwayat penilaian atau catatan kredit sebelumnya dinilai buruk/tidak memenuhi syarat kelayakan minimum.")
                else:
                    st.warning("Alasan Penolakan: Rasio kapasitas total pendapatan pemohon berada di bawah nilai pengajuan pinjaman (Risiko Gagal Bayar Tinggi).")
            
            # 4. Menampilkan Metrik Persentase Berdampingan Sesuai Logika Di Atas
            st.write("###  Analisis Probabilitas Keputusan:")
            metrik_kol1, metrik_kol2 = st.columns(2)
            metrik_kol1.metric(label=" Probabilitas Disetujui (Approved)", value=f"{persen_disetujui:.2f} %")
            metrik_kol2.metric(label=" Probabilitas Ditolak (Rejected)", value=f"{persen_ditolak:.2f} %")
            
            # 5. Menampilkan Ringkasan Teks Monospace Dengan Format Titik Ribuan Indonesia
            credit_status_text = "Good" if cred_enc == 1.0 else "Bad"
            
            app_inc_formatted = f"{int(applicant_income):,}".replace(",", ".")
            co_inc_formatted = f"{int(coapplicant_income):,}".replace(",", ".")
            loan_amt_formatted = f"{int(loan_amount):,}".replace(",", ".")

            summary_text = f"""==================================================

Applicant Summary

Gender              : {gender}
Married             : {married}
Applicant Income    : Rp {app_inc_formatted}
Coapplicant Income  : Rp {co_inc_formatted}
Loan Amount         : Rp {loan_amt_formatted}
Credit History      : {credit_status_text}
Property Area       : {property_area.split(' (')[0]}

=================================================="""
            
            st.write("###  Ringkasan Input Data Pemohon:")
            st.code(summary_text, language="text")
                
        except Exception as error:
            st.error(f"Terjadi kesalahan teknis saat pemrosesan model: {error}")
    else:
        st.error("Proses komputasi gagal dijalankan karena model atau scaler belum siap.")

# ====================================================================
# 6. DOKUMENTASI MATERI EVALUASI & REFLEKSI ETIS (INDIKATOR RTM)
# ====================================================================
st.markdown("---")
