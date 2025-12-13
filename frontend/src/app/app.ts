import { Component, NgZone } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { CommonModule } from '@angular/common';
import { HttpClient, HttpClientModule } from '@angular/common/http';

interface PredictionResponse {
  text: string;
  prediction_class_id: number;
  category_name?: string;
  confidence_scores?: { name: string; value: number }[];
  status: string;
}

interface HealthResponse {
  status: string;
  model_loaded: boolean;
}

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, FormsModule, HttpClientModule],
  templateUrl: './app.html',
  styleUrl: './app.css'
})
export class App {
  title = 'Classification de Texte';
  inputText = '';
  prediction: PredictionResponse | null = null;
  isLoading = false;
  error = '';
  apiStatus: HealthResponse | null = null;

  // Mapping des classes vers des catégories simplifiées
  classNames: { [key: number]: string } = {
    0: 'Religion',
    1: 'Informatique',
    2: 'Informatique',
    3: 'Informatique',
    4: 'Informatique',
    5: 'Informatique',
    6: 'Commerce',
    7: 'Automobile',
    8: 'Automobile',
    9: 'Sport',
    10: 'Sport',
    11: 'Science',
    12: 'Science',
    13: 'Science',
    14: 'Science',
    15: 'Religion',
    16: 'Politique',
    17: 'Politique',
    18: 'Politique',
    19: 'Religion'
  };

  // Icônes pour chaque catégorie
  categoryIcons: { [key: string]: string } = {
    'Religion': '⛪',
    'Informatique': '💻',
    'Commerce': '🛒',
    'Automobile': '🚗',
    'Sport': '⚽',
    'Science': '🔬',
    'Politique': '🏛️'
  };

  private apiUrl = 'http://localhost:8000';

  constructor(private http: HttpClient, private ngZone: NgZone) {
    this.checkApiStatus();
  }

  checkApiStatus(): void {
    this.http.get<HealthResponse>(`${this.apiUrl}/health`).subscribe({
      next: (response) => {
        this.apiStatus = response;
      },
      error: () => {
        this.apiStatus = { status: 'offline', model_loaded: false };
      }
    });
  }

  getClassName(classId: number): string {
    return this.classNames[classId] || `Classe ${classId}`;
  }

  clearForm(): void {
    this.inputText = '';
    this.prediction = null;
    this.error = '';
    this.fileName = '';
  }

  getCategoryIcon(classId: number): string {
    const category = this.getClassName(classId);
    return this.categoryIcons[category] || '📄';
  }

  getConfidenceColor(score: number): string {
    if (score >= 0.7) return 'var(--success-color, #4caf50)'; // Vert
    if (score >= 0.4) return 'var(--warning-color, #ff9800)'; // Orange
    return 'var(--danger-color, #f44336)'; // Rouge
  }

  // File upload
  fileName = '';
  selectedFile: File | null = null;

  onFileSelect(event: Event): void {
    const input = event.target as HTMLInputElement;
    if (!input.files || input.files.length === 0) return;

    const file = input.files[0];
    this.fileName = file.name;
    this.selectedFile = file;

    const allowedTypes = ['.txt', '.md', '.pdf', '.docx'];
    const extension = file.name.substring(file.name.lastIndexOf('.')).toLowerCase();

    if (!allowedTypes.includes(extension)) {
      this.error = 'Format non supporté. Utilisez .txt, .md, .pdf ou .docx';
      return;
    }

    // Pour txt et md, lecture locale
    if (extension === '.txt' || extension === '.md') {
      const reader = new FileReader();
      reader.onload = (e) => {
        this.inputText = e.target?.result as string;
        this.error = '';
        this.selectedFile = null; // Pas besoin d'upload
      };
      reader.onerror = () => {
        this.error = 'Erreur lors de la lecture du fichier.';
      };
      reader.readAsText(file);
    } else {
      // PDF et DOCX seront traités par le backend
      this.inputText = `[Fichier: ${file.name}] - Cliquez sur Analyser pour traiter ce fichier.`;
      this.error = '';
    }
  }

  classify(): void {
    // Si un fichier PDF/DOCX est sélectionné, utiliser l'endpoint upload
    if (this.selectedFile) {
      this.classifyFile();
      return;
    }

    if (!this.inputText.trim()) {
      this.error = 'Veuillez entrer un texte à classifier.';
      return;
    }

    this.isLoading = true;
    this.error = '';
    this.prediction = null;

    this.http.post<PredictionResponse>(`${this.apiUrl}/predict`, { text: this.inputText }).subscribe({
      next: (response) => {
        this.ngZone.run(() => {
          console.log('Response received:', response);
          this.prediction = response;
          this.isLoading = false;
          console.log('isLoading:', this.isLoading, 'prediction:', this.prediction);
        });
      },
      error: (err) => {
        this.ngZone.run(() => {
          console.error('Error:', err);
          this.error = 'Erreur lors de la classification. Vérifiez que l\'API est en ligne.';
          this.isLoading = false;
        });
      }
    });
  }

  classifyFile(): void {
    if (!this.selectedFile) return;

    this.isLoading = true;
    this.error = '';
    this.prediction = null;

    const formData = new FormData();
    formData.append('file', this.selectedFile);

    this.http.post<PredictionResponse>(`${this.apiUrl}/upload`, formData).subscribe({
      next: (response) => {
        this.prediction = response;
        this.isLoading = false;
        this.selectedFile = null;
      },
      error: (err) => {
        this.error = 'Erreur lors de la classification du fichier.';
        this.isLoading = false;
        console.error(err);
      }
    });
  }
}
